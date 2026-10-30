from typing import Any

import torch
from transformers import AutoTokenizer

from train.backend import LocalTrainer
from train.media import model_inputs
from train.models import ModelInput, SamplingParams, TextChunk


def sample(
    trainer: LocalTrainer,
    prompt: list[int] | ModelInput,
    params: SamplingParams,
    count: int = 1,
    prompt_logprobs: bool = False,
    topk: int = 0,
    target_prompt_logprobs: dict | None = None,
) -> dict:
    """Sample local snapshots, returning raw model log probabilities before temperature.

    At most 16 sequences are accepted per request. The prompt and output budget
    must fit the registered model context. String stop sequences require the
    registered model directory to include a tokenizer.
    """
    native_prompt = (
        prompt if isinstance(prompt, ModelInput) else ModelInput(chunks=[TextChunk(tokens=prompt)])
    )
    prompt = native_prompt.tokens()
    model = trainer.model
    context = getattr(model.config.get_text_config(), "max_position_embeddings", 4096)
    budget = params.max_tokens or 16
    if not 0 <= count <= 16 or not prompt or len(prompt) + budget > context:
        raise ValueError("Invalid sample count, empty prompt, or context limit exceeded")
    if any(token < 0 or token >= model.config.get_text_config().vocab_size for token in prompt):
        raise ValueError("Prompt token is out of vocabulary")
    if topk < 0 or topk > model.config.get_text_config().vocab_size:
        raise ValueError("Invalid top-k logprob count")
    generator = torch.Generator(device=trainer.device)
    generator.manual_seed(params.seed) if params.seed is not None else generator.seed()
    stops = _stop_sequences(trainer, params)
    sequences = []
    model.eval()
    with torch.no_grad(), trainer.runtime.autocast(model):
        for _ in range(count):
            sequences.append(_generate(trainer, native_prompt, params, generator, stops, topk))
        scored = None
        if prompt_logprobs or target_prompt_logprobs:
            ids = torch.tensor(prompt, dtype=torch.long, device=trainer.device).unsqueeze(0)
            lp = (
                model(
                    **model_inputs(
                        native_prompt,
                        trainer.spec,
                        trainer.device,
                        model.config,
                        trust_remote_code=trainer.runtime.trust_remote_code,
                    ),
                    use_cache=False,
                )
                .logits[0, :-1]
                .float()
                .log_softmax(-1)
            )
            scored = [None, *lp.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1).tolist()]
    result = {"sequences": sequences, "prompt_logprobs": scored if prompt_logprobs else None}
    if target_prompt_logprobs:
        result["target_prompt_logprobs"] = _targeted_scores(
            lp, target_prompt_logprobs, len(prompt), trainer.device
        )
    return result


def _generate(
    trainer: LocalTrainer,
    prompt: ModelInput,
    params: SamplingParams,
    generator: torch.Generator,
    stops: list[list[int]],
    topk: int,
) -> dict:
    """Sample tokens from the active policy with native request settings."""
    tokens, logprobs, alternatives, sampled_logprobs = [], [], [], []
    reason = "length"
    inputs = model_inputs(
        prompt,
        trainer.spec,
        trainer.device,
        trainer.model.config,
        trust_remote_code=trainer.runtime.trust_remote_code,
    )
    cached = len(inputs) == 1
    past = None
    for _ in range(params.max_tokens or 16):
        output = trainer.model(
            **inputs, use_cache=cached, **({"past_key_values": past} if past is not None else {})
        )
        logits = output.logits[0, -1].float()
        raw = logits.log_softmax(-1)
        token, sampled = _choose(logits, params, generator)
        sampled_logprobs.append(sampled)
        tokens.append(token)
        logprobs.append(float(raw[token]))
        if topk:
            values, indices = raw.topk(topk)
            alternatives.append(list(zip(indices.tolist(), values.tolist(), strict=True)))
        if any(tokens[-len(stop) :] == stop for stop in stops):
            reason = "stop"
            break
        past = output.past_key_values if cached else None
        if past is not None:
            inputs = {"input_ids": torch.tensor([[token]], device=trainer.device)}
        else:
            source = prompt.model_copy(
                update={"chunks": [*prompt.chunks, TextChunk(tokens=tokens)]}
            )
            inputs = model_inputs(
                source,
                trainer.spec,
                trainer.device,
                trainer.model.config,
                trust_remote_code=trainer.runtime.trust_remote_code,
            )
    return {
        "tokens": tokens,
        "logprobs": logprobs,
        "sampling_logprobs": sampled_logprobs,
        "stop_reason": reason,
        "topk_logprobs": alternatives or None,
    }


def _choose(
    logits: torch.Tensor, params: SamplingParams, generator: torch.Generator
) -> tuple[int, float]:
    """Choose the next token under temperature and truncation controls."""
    if params.temperature == 0:
        return int(logits.argmax()), 0.0
    scores = logits / params.temperature
    if params.top_k > 0:
        floor = scores.topk(min(params.top_k, len(scores))).values[-1]
        scores = scores.masked_fill(scores < floor, float("-inf"))
    if params.top_p < 1:
        ordered, indices = scores.sort(descending=True)
        excluded = ordered.softmax(-1).cumsum(-1) - ordered.softmax(-1) >= params.top_p
        scores = scores.scatter(0, indices, ordered.masked_fill(excluded, float("-inf")))
    token = int(torch.multinomial(scores.softmax(-1), 1, generator=generator))
    return token, float(scores.log_softmax(-1)[token])


def _stop_sequences(trainer: LocalTrainer, params: SamplingParams) -> list[list[int]]:
    """Normalize native stop-token and stop-sequence settings."""
    eos = trainer.model.config.get_text_config().eos_token_id
    stops = [[token] for token in (eos if isinstance(eos, list) else [eos]) if token is not None]
    if params.stop is None:
        return stops
    requested = [params.stop] if isinstance(params.stop, str) else params.stop
    if all(isinstance(value, int) for value in requested):
        return stops + [[int(value)] for value in requested]
    tokenizer = AutoTokenizer.from_pretrained(
        trainer.spec.base_path,
        local_files_only=True,
        trust_remote_code=trainer.runtime.trust_remote_code,
    )
    for value in requested:
        if tokenizer is None:
            raise ValueError("String stops require a tokenizer")
        encoded = tokenizer.encode(value, add_special_tokens=False)
        if not encoded:
            raise ValueError("Stop sequence must not be empty")
        stops.append(encoded)
    return stops


def _targeted_scores(
    lp: torch.Tensor, request: dict[str, Any], length: int, device: torch.device
) -> dict[str, Any]:
    """Compute scores for explicitly requested target token IDs."""
    shape = request["shape"]
    requested = torch.tensor(request["data"], dtype=torch.long, device=device).reshape(shape)
    if (
        len(shape) != 2
        or shape[0] != length - 1
        or (requested < -1).any()
        or (requested >= lp.shape[-1]).any()
    ):
        raise ValueError("Invalid request shape or token IDs")
    selected = lp.gather(-1, requested.clamp_min(0)).masked_fill(requested == -1, float("nan"))
    return {"data": selected.flatten().tolist(), "shape": shape, "dtype": "float32"}
