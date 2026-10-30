from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import httpx
from pydantic import Field

from train.checkpoints import Checkpoint
from train.merged import MergedWeights
from train.models import Model, ModelSpec
from train.runtime import Runtime
from train.serving import Serving, locked


class Deployment(Model):
    """An immutable checkpoint binding and its callable model identity."""

    id: str
    base_model: str
    models: dict[str, str] = Field(default_factory=dict)
    addons: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Registry(Model):
    """Persisted deployment bindings and promoted model resources."""

    models: dict[str, str] = Field(default_factory=dict)
    deployments: dict[str, Deployment] = Field(default_factory=dict)


class Deployments:
    """Manage native checkpoint promotion, adapter loading, and deployment bindings."""

    def __init__(
        self, root: Path, models: dict[str, Path], serving: Serving, runtime: Runtime | None = None
    ) -> None:
        """Bind deployment storage to registered models and a serving backend.

        Args:
            root: Directory containing persistent checkpoints and deployment state.
            models: Registered model aliases and their local snapshots.
            serving: Configured inference backend and authentication.
            runtime: Explicit model loading policy for merged checkpoint exports.
        """
        self.root, self.models, self.serving = root, models, serving
        self.path = root / "deployments.json"
        self.merged = MergedWeights(root, runtime or Runtime(serving=serving))

    def registry(self) -> Registry:
        """Load the persisted deployment registry."""
        return Registry.load(self.path) if self.path.exists() else Registry()

    def pin(self, uri: str) -> tuple[str, ModelSpec]:
        """Resolve and pin a deployment to immutable sampler weights."""
        path = Checkpoint.resolve(self.root, uri, self.models)
        if path.parent.name != "sampler_weights":
            raise ValueError("Deployment requires a sampler checkpoint, not optimizer state")
        return f"tinker://{'/'.join(path.relative_to(self.root).parts)}", ModelSpec.load(
            path / "model.json"
        )

    def promote(self, name: str, uri: str, base_model: str | None = None) -> str:
        """Register immutable sampler weights under a provider model resource."""
        with locked(self.root):
            pinned, spec = self.pin(uri)
            if base_model is not None and spec.base_model != base_model:
                raise ValueError("Checkpoint base model does not match promotion")
            registry = self.registry()
            if name in registry.models and registry.models[name] != pinned:
                raise ValueError("Model identity already refers to another snapshot")
            registry.models[name] = pinned
            registry.save(self.path)
            return pinned

    def create(
        self,
        identifier: str,
        bindings: dict[str, str],
        base_model: str,
        addons: bool = False,
        metadata: dict[str, str] | None = None,
    ) -> Deployment:
        """Create a callable deployment for a registered checkpoint."""
        with locked(self.root):
            self._check_base(base_model)
            pinned = {}
            for name, uri in bindings.items():
                pinned[name], spec = self.pin(uri)
                if spec.base_model != base_model:
                    raise ValueError("All deployed adapters must use the deployment base model")
            deployment = Deployment(
                id=identifier,
                base_model=base_model,
                models=pinned,
                addons=addons,
                metadata=metadata or {},
            )
            registry = self.registry()
            if identifier in registry.deployments:
                existing = registry.deployments[identifier]
                if existing.model_dump(exclude={"created_at"}) != deployment.model_dump(
                    exclude={"created_at"}
                ):
                    raise ValueError(
                        "Deployment identity already exists with different configuration"
                    )
                deployment = existing
            with httpx.Client(headers=self.serving.headers, timeout=self.serving.timeout) as client:
                self.check_backend(client, base_model)
                for uri in pinned.values():
                    self.ensure_loaded(client, uri)
            registry.deployments[identifier] = deployment
            registry.save(self.path)
            return deployment

    def attach(self, identifier: str, model: str) -> Deployment:
        """Attach a promoted model to an existing deployment."""
        with locked(self.root):
            registry = self.registry()
            deployment = registry.deployments[identifier]
            if not deployment.addons:
                raise ValueError("Deployment was not created with enableAddons")
            uri = registry.models[model]
            pinned, spec = self.pin(uri)
            if spec.base_model != deployment.base_model:
                raise ValueError("Adapter base model does not match deployment")
            with httpx.Client(headers=self.serving.headers, timeout=self.serving.timeout) as client:
                self.check_backend(client, spec.base_model)
                self.ensure_loaded(client, pinned)
            deployment.models[model] = pinned
            registry.save(self.path)
            return deployment

    def delete(self, identifier: str) -> None:
        """Remove a deployment and unload adapters with no remaining users."""
        with locked(self.root):
            registry = self.registry()
            deployment = registry.deployments[identifier]
            del registry.deployments[identifier]
            retained = {
                uri for item in registry.deployments.values() for uri in item.models.values()
            }
            with httpx.Client(headers=self.serving.headers, timeout=self.serving.timeout) as client:
                response = client.get(self.serving.endpoint("/v1/models"))
                response.raise_for_status()
                loaded = {row["id"] for row in response.json()["data"]}
                for uri in set(deployment.models.values()) - retained:
                    name, _ = self._adapter(uri)
                    if name in loaded:
                        client.post(
                            self.serving.control_endpoint("unload_lora_adapter"),
                            json={"lora_name": name},
                        ).raise_for_status()
            registry.save(self.path)

    def _check_base(self, base_model: str) -> None:
        """Require the checkpoint base model to match the serving backend."""
        configured = self.serving.base_model
        if configured is None and len(self.models) == 1:
            configured = next(iter(self.models))
        if configured != base_model or base_model not in self.models:
            raise ValueError(
                "Set TRAIN_INFERENCE_BASE_MODEL to the registered base model served by the inference backend"
            )

    def check_backend(self, client: httpx.Client, base_model: str) -> None:
        """Require the configured base model before publication."""
        self._check_base(base_model)
        if self.serving.checkpoint_mode == "merged":
            response = client.get(self.serving.endpoint("/get_model_info"))
            response.raise_for_status()
            if "weight_version" not in response.json():
                raise ValueError("SGLang must expose the active checkpoint weight version")
            return
        response = client.get(self.serving.endpoint("/v1/models"))
        response.raise_for_status()
        expected = self.serving.served_model or base_model
        if expected not in {row["id"] for row in response.json()["data"]}:
            raise ValueError("Configured base model is not advertised by the serving backend")

    def _adapter(self, uri: str) -> tuple[str, str]:
        """Resolve the adapter path as mounted inside the serving backend."""
        path = Checkpoint.resolve(self.root, uri, self.models)
        mounted = self.serving.train_root
        remote = mounted / path.relative_to(self.root) if mounted is not None else path
        return "trains-" + sha256(uri.encode()).hexdigest()[:32], str(remote)

    def ensure_loaded(
        self, client: httpx.Client, uri: str | None, base_model: str | None = None
    ) -> str:
        """Select exact checkpoint or base weights through the configured backend."""
        if uri is None:
            if base_model is None:
                raise ValueError("Base model identity is required without a checkpoint")
            spec = ModelSpec(base_model=base_model, base_path=self.models[base_model], rank=1)
        else:
            _, spec = self.pin(uri)
        if self.serving.checkpoint_mode == "merged":
            checkpoint = Checkpoint.resolve(self.root, uri, self.models) if uri else None
            return self.merged.select(client, spec, checkpoint)
        if uri is None:
            return self.serving.served_model or spec.base_model
        name, path = self._adapter(uri)
        response = client.get(self.serving.endpoint("/v1/models"))
        response.raise_for_status()
        if name not in {row["id"] for row in response.json()["data"]}:
            response = client.post(
                self.serving.control_endpoint("load_lora_adapter"),
                json={"lora_name": name, "lora_path": path},
            )
            response.raise_for_status()
            if self.serving.backend == "sglang" and not response.json().get("success"):
                raise ValueError(
                    f"SGLang adapter load failed: {response.json().get('error_message')}"
                )
        return self.serving.adapter_model(spec.base_model, name)

    def resolve(self, model: str, deployment_id: str | None = None) -> tuple[str | None, str]:
        """Resolve a request using only explicitly published bindings or sampler URIs."""
        registry = self.registry()
        if deployment_id is not None:
            if deployment_id not in registry.deployments:
                matches = [
                    name
                    for name in registry.deployments
                    if name.startswith("baseten:") and name.endswith(":" + deployment_id)
                ]
                if len(matches) != 1:
                    raise KeyError("Deployment not found")
                deployment_id = matches[0]
            deployment = registry.deployments[deployment_id]
            if model in deployment.models:
                return deployment.models[model], deployment.base_model
            if not deployment.models and model in {deployment.id, deployment.base_model}:
                return None, deployment.base_model
            raise KeyError("Model is not attached to this deployment")
        if model in registry.deployments:
            deployment = registry.deployments[model]
            if len(deployment.models) == 1:
                return next(iter(deployment.models.values())), deployment.base_model
            if not deployment.models:
                return None, deployment.base_model
            raise ValueError("Specify an adapter model for a multi-LoRA deployment")
        if model in self.models:
            return None, model
        if model.startswith(("tinker://", "bt://", "local/run-")):
            uri, spec = self.pin(model)
            return uri, spec.base_model
        candidates = [d for d in registry.deployments.values() if model in d.models]
        if len(candidates) != 1:
            raise KeyError("Model must have exactly one deployment or use model#deployment")
        return candidates[0].models[model], candidates[0].base_model
