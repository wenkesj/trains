CAPABILITIES = {
    "tinker": frozenset({"text", "image", "audio", "video_frames"}),
    "spindle": frozenset({"text", "image", "video_frames"}),
    "fireworks": frozenset({"text", "image", "audio", "video_frames"}),
    "baseten": frozenset({"text", "image", "video_frames"}),
}


def require_capability(provider: str, capability: str) -> None:
    """Reject a modality not supported by the selected native SDK adapter."""
    if capability not in CAPABILITIES[provider]:
        raise ValueError(f"{provider} does not support {capability} in the pinned native contract")
