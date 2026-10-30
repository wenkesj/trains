from collections.abc import Callable
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from train.attention import chunk_attention


def chunk_mlps(model: torch.nn.Module, size: int) -> None:
    """Replace dense MLP projections with equivalent bounded-memory operations."""
    chunk_attention(model)
    if not size:
        return
    for module in model.modules():
        if isinstance(module.forward, partial) and module.forward.func in {
            _chunked_mlp,
            _chunked_gated_norm,
        }:
            continue
        if type(module).__name__ in {"LlamaMLP", "Qwen3_5MLP", "Lfm3MLP"}:
            module.forward = partial(_chunked_mlp, module.forward, size)
        elif type(module).__name__ == "Qwen3_5RMSNormGated":
            module.forward = partial(_chunked_gated_norm, module.forward, size * 32)


def _chunked_mlp(forward: Callable, size: int, values: torch.Tensor) -> torch.Tensor:
    """Bound pointwise causal-model feed-forward intermediates by sequence chunks.

    All tokens and LoRA parameter names remain unchanged. Inner checkpoints avoid
    retaining a full sequence of expanded gate/up activations during the outer
    layer's backward recomputation.
    """
    if values.shape[-2] <= size:
        return forward(values)
    return torch.cat(
        [
            checkpoint(forward, part, use_reentrant=False)
            if torch.is_grad_enabled() and part.requires_grad
            else forward(part)
            for part in values.split(size, dim=-2)
        ],
        dim=-2,
    )


def _chunked_gated_norm(
    forward: Callable, size: int, values: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Bound tokenwise float32 normalization intermediates without splitting a head."""
    if values.shape[-2] <= size:
        return forward(values, gate)
    return torch.cat(
        [
            checkpoint(forward, part, part_gate, use_reentrant=False)
            if torch.is_grad_enabled() and (part.requires_grad or part_gate.requires_grad)
            else forward(part, part_gate)
            for part, part_gate in zip(
                values.split(size, dim=-2), gate.split(size, dim=-2), strict=True
            )
        ],
        dim=-2,
    )
