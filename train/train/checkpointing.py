from collections.abc import Callable
from typing import Any, Protocol, cast

import torch
from torch.utils.checkpoint import checkpoint


class CheckpointingLayer(Protocol):
    """An activation-checkpointed decoder layer with optional host offload."""

    _gradient_checkpointing_func: Callable[..., Any]


def offload_checkpoints(model: torch.nn.Module) -> None:
    """Move retained checkpoint inputs to host memory during training."""
    for module in model.modules():
        if getattr(module, "gradient_checkpointing", False):
            cast(CheckpointingLayer, module)._gradient_checkpointing_func = _offloaded_checkpoint


def _offloaded_checkpoint[T](function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Keep non-reentrant checkpoint inputs in host memory until recomputation.

    The context surrounds checkpoint creation, so only its retained inputs move
    off GPU; recomputed layer intermediates use checkpoint's own saved-tensor
    hooks. Pageable host buffers avoid retaining a CUDA pinned-memory pool across
    variable-length sequences. Transfers remain synchronous.
    """
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        return checkpoint(function, *args, use_reentrant=False, **kwargs)
