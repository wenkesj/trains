from __future__ import annotations

import fcntl
import gc
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from peft.helpers import disable_input_dtype_casting
from pydantic import Field, PrivateAttr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PreTrainedModel,
)
from transformers.utils.import_utils import is_flash_attn_4_available

from train.kernels import enable_kernels
from train.models import ModelSpec
from train.serving import Serving

if TYPE_CHECKING:
    from train.backend import LocalTrainer


class Runtime(BaseSettings):
    """Registered model loading and local device/kernel configuration."""

    model_config = SettingsConfigDict(env_prefix="TRAIN_")
    device: str = "cpu"
    dtype: Literal["float32", "bfloat16"] = "float32"
    gradient_checkpointing: bool = False
    checkpoint_offload: bool = False
    trust_remote_code: bool = False
    mlp_chunk_size: int = Field(default=1024, ge=0)
    kernels: Literal["auto", "reference"] = "auto"
    attention_backend: Literal["auto", "sdpa", "flash_attention_4"] = "auto"
    sampling_backend: Literal["transformers", "sglang"] = "transformers"
    serving: Serving = Field(default_factory=Serving)
    _kernel_report: dict[str, str] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sampling_backend(self) -> Runtime:
        """Require an explicit SGLang endpoint for native text sampling."""
        if self.sampling_backend == "sglang" and (
            self.serving.backend != "sglang" or not self.serving.url
        ):
            raise ValueError("SGLang token sampling requires a configured SGLang serving endpoint")
        return self

    @property
    def kernel_report(self) -> dict[str, str]:
        """Report the attention and convolution kernels active for this model."""
        return self._kernel_report

    @contextmanager
    def autocast(self, model: torch.nn.Module) -> Generator[None, None, None]:
        """Run CUDA LoRA projections in model precision while retaining FP32 parameters."""
        enabled = self.device.startswith("cuda") and self.dtype == "bfloat16"
        with (
            torch.autocast(
                device_type=torch.device(self.device).type, dtype=torch.bfloat16, enabled=enabled
            ),
            disable_input_dtype_casting(model, active=enabled),
        ):
            yield

    def load(self, spec: ModelSpec) -> PreTrainedModel:
        """Load a registered snapshot and its processor onto the configured backend."""
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; install a CUDA-enabled PyTorch build")
        dtype = getattr(torch, self.dtype)
        if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("The selected GPU does not support bfloat16")
        config = AutoConfig.from_pretrained(
            spec.base_path, local_files_only=True, trust_remote_code=self.trust_remote_code
        )
        loader = (
            AutoModelForImageTextToText
            if hasattr(config, "vision_config")
            else AutoModelForCausalLM
        )
        model: Any = loader.from_pretrained(
            spec.base_path,
            config=config,
            local_files_only=True,
            trust_remote_code=self.trust_remote_code,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        model = self.place(model, device)
        self._kernel_report = enable_kernels(model, self.kernels)
        return model

    def place(self, model: Any, device: str | torch.device) -> Any:
        """Move a model and select its attention backend for that device.

        Args:
            model: Native Transformers model or its PEFT wrapper.
            device: Destination for model weights and attention computation.

        Returns:
            The model using FlashAttention 4 on eligible CUDA devices or SDPA on CPU.
        """
        device = torch.device(device)
        wanted = self._attention_implementation(device)
        if self.attention_backend == "auto" and not getattr(model, "_supports_flash_attn", False):
            wanted = "sdpa"
        model = model.to(device)
        if model.config._attn_implementation != wanted:
            model.set_attn_implementation(_attention_request(model, wanted))
        return model

    def _attention_implementation(self, device: torch.device) -> str:
        """Keep CPU exports and reference execution independent of CUDA attention packages."""
        if device.type != "cuda" or self.kernels == "reference" or self.attention_backend == "sdpa":
            return "sdpa"
        available = self.dtype == "bfloat16" and is_flash_attn_4_available()
        if self.attention_backend == "flash_attention_4" and not available:
            raise ValueError(
                "FlashAttention 4 requires CUDA bfloat16 and trains-train[accelerated]"
            )
        return "flash_attention_4" if available else "sdpa"

    @contextmanager
    def exclusive(
        self, root: Path, engine: Callable[[], LocalTrainer | None]
    ) -> Generator[None, None, None]:
        """Serialize provider SDK operations and offload between RPCs."""
        with (root / ".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            with self.serving.suspended(), self._active(engine):
                yield

    @contextmanager
    def _active(self, engine: Callable[[], LocalTrainer | None]) -> Generator[None, None, None]:
        """Offload all accelerator allocations before waking the inference server."""
        try:
            current = engine()
            if current is not None:
                current.move(self.device)
            yield
        except Exception as exc:
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            current = engine()
            if current is not None and self.device.startswith("cuda"):
                current.move("cpu")
            gc.collect()
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()


def _attention_request(model: Any, wanted: str) -> Any:
    """Select supported native submodels without changing unused auxiliary configs."""
    native = {
        id(module.config)
        for module in model.modules()
        if isinstance(module, PreTrainedModel)
        and (wanted == "sdpa" or getattr(module, "_supports_flash_attn", False))
    }
    native.add(id(model.config))

    def requested(config: Any) -> Any:
        children = {
            name: requested(child)
            for name in getattr(config, "sub_configs", {})
            if (child := getattr(config, name, None)) is not None
        }
        selected = wanted if id(config) in native else config._attn_implementation
        return {"": selected, **children} if children else selected

    return requested(model.config)
