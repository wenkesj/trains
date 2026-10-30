import math
from typing import Literal, TypedDict


class LossTensor(TypedDict):
    """A flattened native loss tensor with its dtype and shape."""

    data: list[int] | list[float]
    dtype: Literal["int64", "float32"]
    shape: list[int]


class EncodedText(TypedDict):
    """A tokenized text model-input chunk."""

    type: Literal["encoded_text"]
    tokens: list[int]


class TextModelInput(TypedDict):
    """An ordered sequence of encoded text chunks."""

    chunks: list[EncodedText]


class TextDatum(TypedDict):
    """Internal wire shape produced by default text adapters; custom adapters are unrestricted."""

    model_input: TextModelInput
    loss_fn_inputs: dict[str, LossTensor]


def encoded_datum(
    prompt: list[int], completion: list[int], advantage: float | None = None
) -> TextDatum:
    """Shift prompt and completion tokens into assistant-masked training inputs.

    Args:
        prompt: Nonempty context token IDs; these positions are masked from loss.
        completion: Nonempty assistant action token IDs to supervise.
        advantage: Optional finite rollout advantage for the assistant positions.

    Returns:
        A next-token training datum with aligned targets and assistant weights.

    Raises:
        ValueError: Token sequences are empty, token IDs invalid, or advantage nonfinite.

    Examples:
        >>> datum = encoded_datum([10, 11], [12, 13])
        >>> datum["loss_fn_inputs"]["target_tokens"]["data"]
        [11, 12, 13]
        >>> datum["loss_fn_inputs"]["weights"]["data"]
        [0.0, 1.0, 1.0]
    """
    if not prompt or not completion:
        raise ValueError("Tokenized prompt and completion must be nonempty")
    tokens = prompt + completion
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Token IDs must be nonnegative integers")
    if advantage is not None and not math.isfinite(advantage):
        raise ValueError("Advantage must be finite")
    prefix = len(prompt) - 1
    loss_inputs: dict[str, LossTensor] = {
        "target_tokens": {"data": tokens[1:], "dtype": "int64", "shape": [len(tokens) - 1]},
        "weights": {
            "data": [0.0] * prefix + [1.0] * len(completion),
            "dtype": "float32",
            "shape": [len(tokens) - 1],
        },
    }
    if advantage is not None:
        loss_inputs["advantages"] = {
            "data": [0.0] * prefix + [advantage] * len(completion),
            "dtype": "float32",
            "shape": [len(tokens) - 1],
        }
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens[:-1]}]},
        "loss_fn_inputs": loss_inputs,
    }


def conversation_datum(tokens: list[int], weights: list[float]) -> TextDatum:
    """Build next-token targets from a fully rendered conversation and its mask."""
    if len(tokens) < 2 or len(tokens) != len(weights) or not any(weights[1:]):
        raise ValueError("Conversation must contain aligned supervised assistant tokens")
    value = encoded_datum(tokens[:1], tokens[1:])
    value["loss_fn_inputs"]["weights"]["data"] = weights[1:]
    return value
