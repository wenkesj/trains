from __future__ import annotations

import gc
import shutil
from hashlib import sha256
from pathlib import Path

import httpx
from peft import PeftModel

from train.models import Model, ModelSpec
from train.runtime import Runtime


class MergedWeights:
    """Select immutable adapters through SGLang's native full-weight reload API."""

    def __init__(self, root: Path, runtime: Runtime) -> None:
        """Use one reusable merged export under the shared training directory.

        Args:
            root: Training checkpoint storage, also mounted in the serving container.
            runtime: Model loading policy and serving configuration.
        """
        self.root = root
        self.runtime = runtime
        self.directory = root / ".serving"

    def select(self, client: httpx.Client, spec: ModelSpec, checkpoint: Path | None) -> str:
        """Load every adapted tensor and verify the backend's resulting weight version.

        Args:
            client: Authenticated backend client; the caller holds the shared GPU lock.
            spec: Registered base model and exact adapter configuration.
            checkpoint: Immutable sampler directory, or None for the original base model.

        Returns:
            The backend's callable model name for the selected checkpoint.
        """
        serving = self.runtime.serving
        identity = str(checkpoint) if checkpoint else "base:" + str(spec.base_path)
        version = "trains-" + sha256(identity.encode()).hexdigest()
        state_path = self.directory / "state.json"
        state = _State.load(state_path) if state_path.exists() else _State()
        response = client.get(serving.endpoint("/get_model_info"))
        response.raise_for_status()
        if state.active != version or response.json().get("weight_version") != version:
            path = self._path(spec, checkpoint, version, state)
            state.active = None
            state.save(state_path)
            response = client.post(
                serving.endpoint("/update_weights_from_disk"),
                json={
                    "model_path": str(path),
                    "load_format": "safetensors",
                    "weight_version": version,
                    "flush_cache": True,
                },
            )
            response.raise_for_status()
            if not response.json().get("success"):
                raise ValueError(f"SGLang weight reload failed: {response.json().get('message')}")
            response = client.get(serving.endpoint("/get_model_info"))
            response.raise_for_status()
            if response.json().get("weight_version") != version:
                raise ValueError("SGLang did not activate the requested checkpoint version")
            state.active = version
            state.save(state_path)
        return response.json().get("served_model_name") or serving.served_model or spec.base_model

    def _path(self, spec: ModelSpec, checkpoint: Path | None, version: str, state: _State) -> Path:
        """Resolve base weights or create the single bounded adapter export."""
        serving = self.runtime.serving
        if checkpoint is None:
            return serving.model_path or spec.base_path
        target = self.directory / "merged"
        cached = any(
            (target / name).is_file()
            for name in ("model.safetensors", "model.safetensors.index.json")
        )
        if state.exported != version or not cached:
            state.exported = None
            state.save(self.directory / "state.json")
            if target.exists():
                shutil.rmtree(target)
            self._export(spec, checkpoint, target)
            state.exported = version
            state.save(self.directory / "state.json")
        mounted = serving.train_root or self.root
        return mounted / target.relative_to(self.root)

    def _export(self, spec: ModelSpec, checkpoint: Path, target: Path) -> None:
        """Merge all adapter projections on CPU without retaining full-model history."""
        runtime = self.runtime.model_copy(update={"device": "cpu"})
        model = PeftModel.from_pretrained(runtime.load(spec), checkpoint).merge_and_unload()
        if not model.generation_config.do_sample:
            # Sampling-only upstream defaults can otherwise invalidate a greedy export.
            model.generation_config.temperature = 1.0
            model.generation_config.top_k = 50
            model.generation_config.top_p = 1.0
        model.save_pretrained(target, max_shard_size="20GB")
        del model
        gc.collect()


class _State(Model):
    exported: str | None = None
    active: str | None = None
