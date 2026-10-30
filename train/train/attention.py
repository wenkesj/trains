from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint


def chunk_attention(model: torch.nn.Module, heads: int = 1) -> None:
    """Bound attention memory with causally equivalent chunked operations."""
    for module in model.modules():
        functions = getattr(module, "_kernel_funcs", {})
        kernel = functions.get("chunk_gated_delta_rule")
        if kernel is None:
            kernel = functions.get("torch_chunk_gated_delta_rule")
        if kernel is not None and not (
            isinstance(kernel.forward, partial) and kernel.forward.func is _head_chunks
        ):
            kernel.forward = partial(_head_chunks, kernel.forward, heads)


def _head_chunks(
    forward: Callable,
    heads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Checkpoint independent GDN heads while preserving every causal token."""
    if (
        q.shape[2] <= heads
        or q.shape[2] != v.shape[2]
        or kwargs.get("initial_state") is not None
        or kwargs.get("output_final_state", False)
        or kwargs.get("cp_context") is not None
    ):
        return forward(q, k, v, g=g, beta=beta, **kwargs)
    outputs = []
    for start in range(0, q.shape[2], heads):
        parts = [x[:, :, start : start + heads] for x in (q, k, v, g, beta)]
        call = partial(forward, **kwargs)
        output, _ = (
            checkpoint(call, *parts, use_reentrant=False)
            if torch.is_grad_enabled() and any(x.requires_grad for x in parts)
            else call(*parts)
        )
        outputs.append(output)
    return torch.cat(outputs, dim=2), None
