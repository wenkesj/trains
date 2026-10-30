import asyncio
import fcntl
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Serving(BaseSettings):
    """The configured inference backend and its authentication."""

    model_config = SettingsConfigDict(env_prefix="TRAIN_INFERENCE_")
    backend: Literal["vllm", "sglang"] = "vllm"
    checkpoint_mode: Literal["lora", "merged"] = "lora"
    model_path: Path | None = None
    url: str | None = None
    api_key: SecretStr | None = None
    train_root: Path | None = None
    base_model: str | None = None
    served_model: str | None = None
    shared_gpu: bool = False
    timeout: float = Field(default=600, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def compatible_checkpoint_mode(self) -> "Serving":
        """Require SGLang's native weight reload API for merged checkpoints."""
        if self.checkpoint_mode == "merged" and self.backend != "sglang":
            raise ValueError("Merged checkpoint serving requires the SGLang backend")
        return self

    @property
    def headers(self) -> dict[str, str]:
        """Return backend authentication headers without forwarding client secrets."""
        return (
            {"Authorization": "Bearer " + self.api_key.get_secret_value()} if self.api_key else {}
        )

    def endpoint(self, path: str) -> str:
        """Resolve an inference route against the configured serving URL."""
        if self.url is None:
            raise ValueError("Configure TRAIN_INFERENCE_URL to enable deployment and chat inference")
        return self.url.rstrip("/") + path

    def control_endpoint(
        self, operation: Literal["load_lora_adapter", "unload_lora_adapter"]
    ) -> str:
        """Resolve the backend's native adapter-management route."""
        prefix = "/v1/" if self.backend == "vllm" else "/"
        return self.endpoint(prefix + operation)

    def adapter_model(self, base: str, adapter: str) -> str:
        """Select the loaded adapter using the backend's model identifier format."""
        if self.backend == "sglang":
            return f"{self.served_model or base}:{adapter}"
        return adapter

    @contextmanager
    def suspended(self) -> Generator[None, None, None]:
        """Release inference cache memory while preserving the served model weights."""
        if not self.shared_gpu:
            yield
            return
        with httpx.Client(headers=self.headers, timeout=self.timeout) as client:
            try:
                if self.backend == "sglang":
                    client.post(
                        self.endpoint("/release_memory_occupation"), json={"tags": ["kv_cache"]}
                    ).raise_for_status()
                else:
                    client.post(self.endpoint("/sleep"), params={"level": 1}).raise_for_status()
                yield
            finally:
                if self.backend == "sglang":
                    client.post(
                        self.endpoint("/resume_memory_occupation"), json={"tags": ["kv_cache"]}
                    ).raise_for_status()
                else:
                    client.post(self.endpoint("/wake_up")).raise_for_status()


@contextmanager
def locked(root: Path) -> Generator[None, None, None]:
    """Serialize inference and training access to a shared GPU."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


@asynccontextmanager
async def async_locked(root: Path) -> AsyncGenerator[None, None]:
    """Acquire without blocking the event loop, including cancellation while waiting."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.01)
        yield
