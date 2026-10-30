import math
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class Model(BaseModel):
    """A local LoRA model and its native training configuration."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def load(cls, path: Path) -> Self:
        """Restore persisted model configuration."""
        return cls.model_validate_json(path.read_text())

    def save(self, path: Path) -> None:
        """Persist model configuration for session restoration."""
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(path.suffix + ".pending")
        pending.write_text(self.model_dump_json(indent=2) + "\n")
        pending.replace(path)


class Tensor(Model):
    """A flattened typed tensor carried by native SDK requests."""

    data: list[FiniteFloat | int]
    dtype: Literal["int64", "float32"]
    shape: list[int] | None = None

    @model_validator(mode="after")
    def vector(self) -> Self:
        """Validate the tensor as a one-dimensional aligned loss vector."""
        if self.shape is not None and (
            not self.shape
            or any(d < 0 for d in self.shape)
            or math.prod(self.shape) != len(self.data)
        ):
            raise ValueError("Dense tensor shape must match its flat data")
        if self.dtype == "int64" and any(int(v) != v for v in self.data):
            raise ValueError("Integer tensors require integer values")
        return self


class TextChunk(Model):
    """An encoded text chunk with explicit model tokens."""

    type: Literal["encoded_text"] = "encoded_text"
    tokens: list[int]


class ImageChunk(Model):
    """An image payload and its expanded model-position count."""

    type: Literal["image"] = "image"
    data: str
    format: Literal["png", "jpeg"]
    expected_tokens: int = Field(gt=0)


class AudioChunk(Model):
    """A native DMel audio payload preserved at the protocol boundary."""

    type: Literal["dmel"] = "dmel"
    dmel: str


class ModelInput(Model):
    """Ordered text and media chunks supplied to a model."""

    chunks: list[TextChunk | ImageChunk | AudioChunk]

    def tokens(self) -> list[int]:
        """Flatten encoded text chunks into their token sequence."""
        result = []
        for chunk in self.chunks:
            if isinstance(chunk, TextChunk):
                result.extend(chunk.tokens)
            elif isinstance(chunk, ImageChunk):
                result.extend([0] * chunk.expected_tokens)
            else:
                raise ValueError("The local backend cannot decode native DMel audio")
        return result


class Datum(Model):
    """A model input and its aligned native loss tensors."""

    model_input: ModelInput
    loss_fn_inputs: dict[str, Tensor]
    model_input_spans: list[dict] | None = None
    loss_fn_input_spans: list[dict] | None = None

    @model_validator(mode="before")
    @classmethod
    def text_only(cls, value: Any) -> Any:
        """Return whether the datum contains only encoded text chunks."""
        if isinstance(value, dict) and "routed_experts_ref" in value:
            if value["routed_experts_ref"] is not None:
                raise ValueError("Router replay is not supported locally")
            return {key: item for key, item in value.items() if key != "routed_experts_ref"}
        return value


class Batch(Model):
    """A training batch and its loss configuration."""

    data: list[Datum] = Field(min_length=1)
    loss_fn: Literal["cross_entropy", "importance_sampling", "ppo"] = "cross_entropy"
    loss_fn_config: dict[str, FiniteFloat | str] | None = None


class ModelSpec(Model):
    """A registered model alias and its local snapshot directory."""

    base_model: str
    base_path: Path
    rank: int = Field(ge=1)
    alpha: int | None = Field(default=None, ge=1)
    seed: int = 0
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = True


class AdamParams(Model):
    """Native AdamW learning-rate and optimizer hyperparameters."""

    learning_rate: FiniteFloat = Field(default=1e-4, gt=0)
    beta1: FiniteFloat = Field(default=0.9, ge=0, lt=1)
    beta2: FiniteFloat = Field(default=0.95, ge=0, lt=1)
    eps: FiniteFloat = Field(default=1e-12, gt=0)
    weight_decay: FiniteFloat = Field(default=0, ge=0)
    grad_clip_norm: FiniteFloat = Field(default=0, ge=0)


class SamplingParams(Model):
    """Token-sampling controls and stop conditions."""

    max_tokens: int | None = Field(default=16, ge=1)
    seed: int | None = None
    temperature: FiniteFloat = Field(default=1, ge=0)
    top_k: int = -1
    top_p: FiniteFloat = Field(default=1, gt=0, le=1)
    stop: list[int] | list[str] | str | None = None
