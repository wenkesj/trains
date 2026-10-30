from collections.abc import Mapping
from typing import Any


def sampling_parameters(request: Mapping[str, Any]) -> dict[str, Any]:
    """Retain the sampling controls needed to validate behavior-policy replay."""
    values = {**request, **(request.get("extra_body") or {})}
    return {key: values.get(key) for key in ("temperature", "top_p", "top_k", "min_p")}
