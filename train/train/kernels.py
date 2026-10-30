import logging
from typing import Any

from transformers import KernelConfig

log = logging.getLogger(__name__)

MAMBA_REVISION = "551d8a26cd19dcb18e52649bdb5c4f1c990b42ac"
FLA_REVISION = "24d8640db48c804a224b5da987b81093ff7136d1"


def enable_kernels(model: Any, policy: str) -> dict[str, str]:
    """Select backward-capable hybrid kernels; explicit reference stays offline.

    The convolution snapshot includes the torch 2.11/CUDA 13 x86-64 build used by
    the training image. Pin revisions so package upgrades cannot change kernels.
    Other architectures retain PyTorch SDPA, which selects CUDA attention kernels.
    """
    attention = model.config.get_text_config()._attn_implementation or "sdpa"
    if model.config.get_text_config().model_type == "lfm3":
        return {"attention": attention, "convolution": "torch-depthwise-width-7"}
    report = {"attention": attention, "linear_attention": "reference", "convolution": "reference"}
    if model.device.type != "cuda" or policy == "reference":
        return report
    config = model.config.get_text_config()
    if not any(value == "linear_attention" for value in getattr(config, "layer_types", [])):
        return {"attention": attention}
    mapping = {
        name: (f"kernels-community/mamba-ssm:{name}", {"revision": MAMBA_REVISION})
        for name in ("causal_conv1d_fn", "causal_conv1d_update")
    }
    mapping["chunk_gated_delta_rule"] = (
        "kernels-community/fla:chunk_gated_delta_rule",
        {"revision": FLA_REVISION},
    )
    model.train()
    model.set_use_kernels(True, KernelConfig(kernel_mapping=mapping, inherit_mapping=False))
    report.update(linear_attention="fla:" + FLA_REVISION, convolution="mamba-ssm:" + MAMBA_REVISION)
    log.info("Training kernel implementation: %s", report)
    return report
