import inspect
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights
from torch.utils.checkpoint import checkpoint
from transformers.pytorch_utils import Conv1D

from train.checkpointing import offload_checkpoints
from train.media import model_inputs
from train.mlp import chunk_mlps
from train.models import AdamParams, Batch, Datum, ModelSpec
from train.objectives import objective, validate_batch
from train.runtime import Runtime


class LocalTrainer:
    """A local causal model with trainable LoRA adapters and persistent Adam state."""

    def __init__(
        self, spec: ModelSpec, adapter: Path | None = None, runtime: Runtime | None = None
    ) -> None:
        """Load the selected base model and LoRA adapter into the local runtime.

        Args:
            spec: Registered base-model identity and local snapshot.
            adapter: Saved LoRA model configuration.
            runtime: Local model-loading, device, and kernel configuration.
        """
        self.spec = spec
        self.runtime = runtime or Runtime()
        base = self.runtime.load(spec)
        self.kernel_report = dict(self.runtime.kernel_report)
        self.device = base.device
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            torch.manual_seed(spec.seed)
            self.model = (
                PeftModel.from_pretrained(base, adapter, is_trainable=False)
                if adapter
                else get_peft_model(
                    base,
                    LoraConfig(
                        r=spec.rank,
                        lora_alpha=spec.alpha or spec.rank,
                        lora_dropout=0,
                        target_modules=_lora_targets(base, spec),
                        task_type="CAUSAL_LM",
                    ),
                )
            )
        chunk_mlps(self.model, self.runtime.mlp_chunk_size)
        if self.runtime.gradient_checkpointing and adapter is None:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.enable_input_require_grads()
            if self.runtime.checkpoint_offload:
                offload_checkpoints(self.model)
            self.kernel_report["activation_checkpointing"] = (
                "cpu_offload" if self.runtime.checkpoint_offload else "device"
            )
        self.model.eval()
        chunk_mlps(self.model, self.runtime.mlp_chunk_size)
        self.step = 0
        self.optimizer = (
            torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0
            )
            if adapter is None
            else None
        )
        self.params = AdamParams()
        self.loss_tokens = 0
        self.sequences = 0

    def move(self, device: str) -> None:
        """Move the active model and optimizer state to the requested device."""
        self.device = torch.device(device)
        self.runtime.place(self.model, self.device)
        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and key != "step":
                        state[key] = value.to(self.device)

    def forward(self, batch: Batch, backward: bool, return_outputs: bool = True) -> dict:
        """Score or backpropagate a batch under the requested native loss."""
        config = self.model.config.get_text_config()
        context = getattr(config, "max_position_embeddings", 4096)
        validate_batch(batch, config.vocab_size, context)
        outputs, losses = [], []
        self.model.train(backward and self.runtime.gradient_checkpointing)
        chunk_mlps(self.model, self.runtime.mlp_chunk_size)
        with self._gradients(backward):
            for datum in batch.data:
                loss, output = self._datum(datum, batch, backward, return_outputs)
                losses.append(loss)
                outputs.append(output)
            if backward:
                self.loss_tokens += sum(
                    sum(value != 0 for value in d.loss_fn_inputs["weights"].data)
                    if "weights" in d.loss_fn_inputs
                    else len(d.model_input.tokens())
                    for d in batch.data
                )
                self.sequences += len(batch.data)
        total = sum(len(d.model_input.tokens()) for d in batch.data)
        return {
            "loss": sum(losses),
            "loss_fn_output_type": "ArrayRecord",
            "loss_fn_outputs": outputs,
            "metrics": {
                "loss:sum": sum(losses),
                **(
                    {"skip_token_normalization": 1.0}
                    if (batch.loss_fn_config or {}).get("skip_token_normalization")
                    else {}
                ),
                "loss:mean": sum(losses) / total,
                "num_tokens:sum": float(total),
            },
        }

    @contextmanager
    def _gradients(self, backward: bool) -> Generator[None, None, None]:
        if not backward:
            yield
            return
        saved = [
            (parameter, parameter.grad.clone() if parameter.grad is not None else None)
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        counts = (self.loss_tokens, self.sequences)
        try:
            yield
        except Exception:
            for parameter, gradient in saved:
                parameter.grad = gradient
            self.loss_tokens, self.sequences = counts
            raise

    def _datum(
        self, datum: Datum, batch: Batch, backward: bool, return_outputs: bool
    ) -> tuple[float, dict]:
        temperature = float((batch.loss_fn_config or {}).get("temperature", 1))
        with torch.set_grad_enabled(backward), self.runtime.autocast(self.model):
            inputs = model_inputs(
                datum.model_input,
                self.spec,
                self.device,
                self.model.config,
                trust_remote_code=self.runtime.trust_remote_code,
            )
            targets = torch.tensor(
                datum.loss_fn_inputs["target_tokens"].data, dtype=torch.long, device=self.device
            )
            targets = targets.reshape(datum.loss_fn_inputs["target_tokens"].shape or [len(targets)])
            selected = (
                list(range(len(targets))) if return_outputs else _loss_positions(datum, batch)
            )
            if not selected:
                return 0.0, {}
            indices = torch.tensor(selected, device=self.device)
            logprobs = _selected_logprobs(
                self.model, inputs, indices, targets[indices], temperature
            )
            loss = objective(
                logprobs, datum if return_outputs else _select_datum(datum, selected), batch
            )
            if not torch.isfinite(loss):
                raise ValueError("Loss is not finite")
            if backward:
                if self.optimizer is None:
                    raise ValueError("Sampler snapshots cannot train")
                loss.backward()
            if not return_outputs:
                return float(loss.detach()), {}
            output = {
                "logprobs": {
                    "data": logprobs.detach().flatten().tolist(),
                    "dtype": "float32",
                    "shape": list(targets.shape),
                }
            }
            return float(loss.detach()), output

    def optim_step(self, params: AdamParams, normalization: str = "none") -> dict:
        """Apply AdamW to accumulated gradients and advance optimizer state."""
        if self.optimizer is None or self.sequences == 0:
            raise ValueError("No accumulated gradients")
        scale = {
            "none": 1,
            "num_loss_tokens": self.loss_tokens,
            "num_sequences": self.sequences,
        }.get(normalization)
        if scale is None or scale == 0:
            raise ValueError("Unsupported gradient normalization")
        for group in self.optimizer.param_groups:
            group.update(
                lr=params.learning_rate,
                betas=(params.beta1, params.beta2),
                eps=params.eps,
                weight_decay=params.weight_decay,
            )
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(scale)
        if params.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), params.grad_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.loss_tokens = self.sequences = 0
        self.params = params
        self.step += 1
        return {"metrics": {"step": float(self.step)}}

    def save(self, path: Path, sampler: bool) -> None:
        """Persist model, optimizer, and pending training state."""
        path.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(str(path))
        self.spec.save(path / "model.json")
        if not sampler:
            if self.optimizer is None:
                raise ValueError("Sampler snapshots have no optimizer")
            gradients = {n: p.grad for n, p in self.model.named_parameters() if p.grad is not None}
            torch.save(
                {
                    "optimizer": self.optimizer.state_dict(),
                    "step": self.step,
                    "gradients": gradients,
                    "loss_tokens": self.loss_tokens,
                    "sequences": self.sequences,
                    "params": self.params.model_dump(),
                },
                path / "trainer.pt",
            )

    def load(self, path: Path, optimizer: bool) -> dict:
        """Restore a saved model and optionally its optimizer state."""
        spec = ModelSpec.load(path / "model.json")
        fields = ("base_path", "rank", "alpha", "train_attn", "train_mlp", "train_unembed")
        if any(getattr(spec, field) != getattr(self.spec, field) for field in fields):
            raise ValueError("Checkpoint belongs to a different model configuration")
        saved = (
            torch.load(path / "trainer.pt", map_location="cpu", weights_only=True)
            if optimizer
            else None
        )
        self.model.zero_grad(set_to_none=True)
        state = load_peft_weights(str(path), device="cpu")
        set_peft_model_state_dict(self.model, state)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0
        )
        self.step = self.loss_tokens = self.sequences = 0
        if saved is not None:
            self.optimizer.load_state_dict(saved["optimizer"])
            self.step, self.loss_tokens, self.sequences = (
                saved["step"],
                saved["loss_tokens"],
                saved["sequences"],
            )
            self.params = AdamParams.model_validate(saved["params"])
            for name, parameter in self.model.named_parameters():
                gradient = saved["gradients"].get(name)
                parameter.grad = gradient.to(self.device) if gradient is not None else None
        return {"step": self.step}


def _lora_targets(model: Any, spec: ModelSpec) -> list[str]:
    """Select trainable projection modules for the configured LoRA policy."""
    targets = []
    output = model.get_output_embeddings()
    for name, layer in model.named_modules():
        if model.config.model_type == "lfm3_omni" and not (
            name.startswith("model.language_model.") or layer is output
        ):
            continue
        if any(
            part in name
            for part in ("visual", "vision_tower", "vision_model", "audio_tower", "audio_model")
        ):
            continue
        if not isinstance(layer, torch.nn.Linear | Conv1D):
            continue
        if layer is output:
            enabled = spec.train_unembed
        elif "attn" in name or "attention" in name:
            enabled = spec.train_attn
        elif any(part in name for part in ("mlp", "ffn", "feed_forward")):
            enabled = spec.train_mlp
        else:
            enabled = spec.train_attn and spec.train_mlp
        if enabled:
            targets.append(name)
    if not targets:
        raise ValueError("LoRA configuration selects no supported linear layers")
    return targets


def _sampled_logprobs(
    logits: torch.Tensor, targets: torch.Tensor, temperature: float = 1
) -> torch.Tensor:
    """Score tokens in 128-position chunks, recomputing float32 softmax in backward.

    Checkpointing retains the original model-dtype logits without retaining full
    sequence-by-vocabulary float32 softmax tensors or their backward buffers.
    """

    def score(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return (
            (values.float() / temperature)
            .log_softmax(-1)
            .gather(-1, labels.unsqueeze(-1))
            .reshape(labels.shape)
        )

    return torch.cat(
        [
            checkpoint(score, values, labels, use_reentrant=False)
            if torch.is_grad_enabled() and logits.requires_grad
            else score(values, labels)
            for values, labels in zip(logits.split(128), targets.split(128), strict=True)
        ]
    )


def _loss_positions(datum: Datum, batch: Batch) -> list[int]:
    """Output-free backend calls score only targets with nonzero weights and advantages.

    All context tokens still enter the transformer. Gradient normalization counts
    the original batch's loss tokens, so removing zero terms does not rescale it.
    SDK forward/backward calls retain complete token log probabilities.
    """
    count = len(datum.model_input.tokens())
    weights = datum.loss_fn_inputs.get("weights")
    advantages = (
        datum.loss_fn_inputs.get("advantages") if batch.loss_fn != "cross_entropy" else None
    )
    return [
        i
        for i in range(count)
        if (weights is None or weights.data[i] != 0)
        and (advantages is None or advantages.data[i] != 0)
    ]


def _select_datum(datum: Datum, indices: list[int]) -> Datum:
    """Extract validated text targets and loss tensors from a native datum."""
    return datum.model_copy(
        update={
            "loss_fn_inputs": {
                name: value.model_copy(
                    update={"data": [value.data[i] for i in indices], "shape": [len(indices)]}
                )
                for name, value in datum.loss_fn_inputs.items()
            }
        }
    )


def _selected_logits(
    model: Any, inputs: dict[str, torch.Tensor], indices: torch.Tensor
) -> torch.Tensor:
    """Project only requested token positions through the model output head."""
    supports_selection = (
        "logits_to_keep" in inspect.signature(model.get_base_model().forward).parameters
    )
    kwargs = {"logits_to_keep": indices} if supports_selection else {}
    logits = model(**inputs, use_cache=False, **kwargs).logits[0]
    return logits if supports_selection else logits[indices]


def _selected_logprobs(
    model: Any,
    inputs: dict[str, torch.Tensor],
    indices: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Bound supported causal-model vocabulary activations to 128 completion positions.

    These architectures apply a plain linear LM head to decoder states. Recompute
    that head and softmax per chunk in backward, retaining hidden states rather
    than a completion-by-vocabulary logits tensor. Other models retain their own
    forward path, including any model-specific logit transforms.
    """
    base = model.get_base_model()
    if len(inputs) > 1 or base.config.get_text_config().model_type not in {
        "llama",
        "qwen3_5_text",
        "lfm3",
    }:
        return _sampled_logprobs(_selected_logits(model, inputs, indices), targets, temperature)
    hidden = base.get_decoder()(**inputs, use_cache=False).last_hidden_state[0][indices]
    head = model.get_output_embeddings()

    def score(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return (
            (head(values).float() / temperature)
            .log_softmax(-1)
            .gather(-1, labels.unsqueeze(-1))
            .reshape(labels.shape)
        )

    return torch.cat(
        [
            checkpoint(score, values, labels, use_reentrant=False)
            if torch.is_grad_enabled() and hidden.requires_grad
            else score(values, labels)
            for values, labels in zip(hidden.split(128), targets.split(128), strict=True)
        ]
    )
