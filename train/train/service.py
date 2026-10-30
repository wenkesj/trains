import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from train.backend import LocalTrainer
from train.checkpoints import Checkpoint
from train.deployments import Deployments
from train.models import AdamParams, Batch, ModelInput, ModelSpec, SamplingParams
from train.native_sampling import sample as sample_native
from train.native_sampling import supports as supports_native
from train.runtime import Runtime
from train.sampling import sample


@dataclass
class Operation:
    """A submitted native operation and its asynchronous result state."""

    id: str
    kind: str
    payload: dict
    result: dict | None = None
    error: str | None = None
    done: bool = False


@dataclass
class TrainingRun:
    """A local trainer with model identity and pending operations."""

    id: str
    spec: ModelSpec
    engine: LocalTrainer | None = None
    cached_state: Path | None = None
    protocol: str = "tinker"
    session_id: str = ""
    next_sequence: int = 0
    pending: dict[int, Operation] = field(default_factory=dict)
    history: dict[int, Operation] = field(default_factory=dict)
    task: asyncio.Task | None = None
    published: str | None = None


class TrainingService:
    """Ordered local training operations with idempotency by model and sequence ID.

    One executor serializes model work. Public futures remain retrievable until
    process exit; model sessions are in memory and checkpoints persist on disk.
    """

    def __init__(self, models: dict[str, Path], root: Path, runtime: Runtime | None = None) -> None:
        """Register local models, persistent storage, and the runtime configuration.

        Args:
            models: Registered model aliases and their local snapshots.
            root: Directory containing persistent checkpoints and deployment state.
            runtime: Local model-loading, device, and kernel configuration.
        """
        self.runtime = runtime or Runtime()
        self.sample_requests: dict[str, tuple[dict, Operation]] = {}
        self.models = {name: path.resolve() for name, path in models.items()}
        self.root = root.resolve()
        self.deployments = Deployments(self.root, self.models, self.runtime.serving, self.runtime)
        self.sessions: set[str] = set()
        self.runs: dict[str, TrainingRun] = {}
        self.operations: dict[str, Operation] = {}
        self.samplers: dict[str, str] = {}
        self.sampler_engines: dict[str, LocalTrainer] = {}
        self.tasks: set[asyncio.Task] = set()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train")

    def create_session(self) -> str:
        """Register a new native client session."""
        session = uuid4().hex
        self.sessions.add(session)
        return session

    def create_model(self, body: dict) -> Operation:
        """Create a LoRA training run for a registered base model."""
        if body.get("session_id") not in self.sessions:
            raise KeyError("Session not found")
        base = body["base_model"]
        config = body.get("lora_config") or {}
        rank = config.get("rank", body.get("lora_rank", 0))
        if rank <= 0 or body.get("optimizer_config", {}).get("type", "adam") not in {
            "adam",
            "adamw",
        }:
            raise ValueError("Only positive-rank LoRA with Adam is supported")
        if base not in self.models:
            raise ValueError("Base model must be registered with --model name=/local/path")
        protocol = body.get("_protocol") or (
            "fireworks" if "alpha" in config or "init_method" in config else "tinker"
        )
        if protocol not in {"fireworks", "tinker"}:
            raise ValueError("Unsupported local training protocol")
        if config.get("init_method") not in {None, "kaiming"}:
            raise ValueError("Only Kaiming LoRA initialization is supported")
        config = {key: value for key, value in config.items() if key != "init_method"}
        if protocol == "fireworks" and config.get("alpha") is None:
            config["alpha"] = 2 * rank
        spec = ModelSpec(
            base_model=base,
            base_path=self.models[base],
            rank=rank,
            **{key: value for key, value in config.items() if key != "rank" and value is not None},
        )
        slot = (body["session_id"], body.get("model_seq_id", len(self.runs)))
        identifier = uuid5(NAMESPACE_URL, f"trains:{slot}").hex
        if identifier in self.runs:
            if self.runs[identifier].spec != spec or self.runs[identifier].protocol != protocol:
                raise ValueError("Model sequence ID reused with different configuration")
            return self.runs[identifier].history[0]
        run = TrainingRun(identifier, spec, protocol=protocol, session_id=body["session_id"])
        self.runs[identifier] = run
        return self.submit(identifier, 0, "create", {})

    def submit(self, model_id: str, seq: int | None, kind: str, payload: dict) -> Operation:
        """Execute a native training operation with deduplication."""
        run = self.runs[model_id]
        if run.session_id not in self.sessions:
            raise ValueError("Training session is finished")
        if seq is None or seq < 0:
            raise ValueError("A nonnegative sequence ID is required")
        if seq in run.history:
            previous = run.history[seq]
            if previous.kind != kind or previous.payload != payload:
                raise ValueError("Sequence ID reused with different operation content")
            return previous
        operation = Operation(uuid4().hex, kind, payload)
        run.history[seq] = run.pending[seq] = operation
        self.operations[operation.id] = operation
        if run.task is None or run.task.done():
            run.task = asyncio.create_task(self._drain(run))
            self.tasks.add(run.task)
            run.task.add_done_callback(self.tasks.discard)
        return operation

    def checkpoint(self, uri: str) -> Path:
        """Save or restore model state through a native checkpoint operation."""
        return Checkpoint.resolve(self.root, uri, self.models)

    def create_sampler(self, uri: str | None, base_model: str | None = None) -> str:
        """Register a sampler pinned to an immutable checkpoint."""
        if uri is None:
            if base_model not in self.models:
                raise ValueError("Unknown base model")
            uri = f"base://{base_model}"
        else:
            path = self.checkpoint(uri)
            uri = f"tinker://{'/'.join(path.relative_to(self.root).parts)}"
        identifier = uuid4().hex
        self.samplers[identifier] = uri
        return identifier

    def sample(self, body: dict, key: str | None = None, source: dict | None = None) -> Operation:
        """Generate a native token-sampling response from a pinned policy."""
        if key is None and body.get("sampling_session_id") and body.get("seq_id") is not None:
            key = f"session:{body['sampling_session_id']}:{body['seq_id']}"
        if key and key in self.sample_requests:
            previous, operation = self.sample_requests[key]
            if previous != (source if source is not None else body):
                raise ValueError("Sampling operation ID reused with different input")
            return operation
        uri = body.get("model_path")
        if not uri:
            if body.get("sampling_session_id"):
                uri = self.samplers[body["sampling_session_id"]]
            elif body.get("base_model") in self.models:
                uri = f"base://{body['base_model']}"
            else:
                raise ValueError("Sampling requires a registered base model or checkpoint")
        if not uri.startswith("base://"):
            path = self.checkpoint(uri)
            uri = f"tinker://{'/'.join(path.relative_to(self.root).parts)}"
        operation = Operation(uuid4().hex, "sample", {**body, "model_path": uri})
        if key:
            self.sample_requests[key] = (source if source is not None else body, operation)
        self.operations[operation.id] = operation
        task = asyncio.create_task(self._sample(operation))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return operation

    async def close(self) -> None:
        """Close a client session and release its active resources."""
        if self.tasks:
            await asyncio.gather(*self.tasks)
        self.executor.shutdown(wait=True)

    async def _drain(self, run: TrainingRun) -> None:
        while run.next_sequence in run.pending:
            operation = run.pending.pop(run.next_sequence)
            try:
                operation.result = await asyncio.get_running_loop().run_in_executor(
                    self.executor, self._execute, run, operation
                )
            except Exception as exc:
                operation.error = f"{type(exc).__name__}: {exc}"
            finally:
                operation.done = True
                run.next_sequence += 1

    async def _sample(self, operation: Operation) -> None:
        try:
            operation.result = await asyncio.get_running_loop().run_in_executor(
                self.executor, self._generate, operation.payload
            )
        except Exception as exc:
            operation.error = f"{type(exc).__name__}: {exc}"
        finally:
            operation.done = True

    def _generate(self, body: dict) -> dict:
        if (
            body.get("topk_prompt_logprobs")
            or body.get("prompt_alt_tokens_k")
            or body.get("prompt_logprobs_last_n") is not None
        ):
            raise ValueError("Advanced prompt scoring is not supported locally")
        if self.runtime.sampling_backend == "sglang" and supports_native(body):
            return sample_native(self, body)
        uri = body["model_path"]
        with self.runtime.exclusive(self.root, lambda: self.sampler_engines.get(uri)):
            if uri not in self.sampler_engines:
                self.sampler_engines.clear()
                if uri.startswith("base://"):
                    name = uri.removeprefix("base://")
                    spec = ModelSpec(
                        base_model=name, base_path=self.models[name], rank=1, train_unembed=False
                    )
                    self.sampler_engines[uri] = LocalTrainer(spec, runtime=self.runtime)
                else:
                    path = self.checkpoint(uri)
                    self.sampler_engines[uri] = LocalTrainer(
                        ModelSpec.load(path / "model.json"), adapter=path, runtime=self.runtime
                    )
            return sample(
                self.sampler_engines[uri],
                ModelInput.model_validate(body["prompt"]),
                SamplingParams.model_validate(body.get("sampling_params", {})),
                body.get("num_samples", 1),
                body.get("prompt_logprobs", False),
                body.get("topk_sample_logprobs", 0),
                body.get("target_prompt_logprobs"),
            )

    def _execute(self, run: TrainingRun, operation: Operation) -> dict[str, Any]:
        with self.runtime.exclusive(self.root, lambda: run.engine):
            _resident(self, run)
            return _execute(self, run, operation)


def _resident(service: TrainingService, active: TrainingRun) -> None:
    """Keep one trainer resident; preserve evicted sessions including pending gradients.

    Closing a native SDK transport need not terminate its remote trainer.
    Disk snapshots bound local host memory without adding lifecycle endpoints.
    The single executor owns all cache transitions.
    """
    for run in service.runs.values():
        if run is active or run.engine is None:
            continue
        path = service.root / run.id / "resident" / uuid4().hex
        run.engine.save(path, sampler=False)
        run.cached_state = path
        run.engine = None
    gc.collect()
    if active.engine is None and active.cached_state is not None:
        active.engine = LocalTrainer(active.spec, runtime=service.runtime)
        active.engine.load(active.cached_state, optimizer=True)


def _execute(service: TrainingService, run: TrainingRun, operation: Operation) -> dict[str, Any]:
    body, kind = operation.payload, operation.kind
    if kind == "create":
        run.engine = LocalTrainer(run.spec, runtime=service.runtime)
        return {"model_id": run.id, "type": "create_model"}
    engine = run.engine
    if engine is None:
        raise ValueError("Model creation failed")
    if kind in {"forward", "forward_backward"}:
        return engine.forward(
            Batch.model_validate(body["forward_backward_input"]), kind == "forward_backward"
        )
    if kind == "optim_step":
        return engine.optim_step(
            AdamParams.model_validate(body.get("adam_params", body.get("optim_params", {}))),
            body.get("grad_accumulation_normalization") or "none",
        )
    if kind == "init":
        return {"step": engine.step, "lora_rank": run.spec.rank}
    if kind in {"save_weights", "save_weights_for_sampler"}:
        sampler = kind == "save_weights_for_sampler"
        record = Checkpoint.publish(
            service.root,
            run.id,
            body.get("path") or body.get("name") or uuid4().hex,
            operation.id,
            engine,
            sampler,
            bool(body.get("overwrite") or body.get("loops")),
        )
        if sampler:
            run.published = record.uri
        result = {"path": record.uri, "version": engine.step, "type": kind}
        if sampler and body.get("sampling_session_seq_id") is not None:
            result["sampling_session_id"] = service.create_sampler(record.uri)
        if run.protocol == "fireworks":
            result["path"] = f"local/run-{run.id}/{record.directory.name}"
        return result
    if kind == "load_weights":
        return engine.load(service.checkpoint(body["path"]), bool(body.get("optimizer")))
    raise ValueError(f"Unsupported local operation: {kind}")
