import json
import math
from collections.abc import Mapping
from typing import Any

from datasets import Features, Value
from trains.types import Provenance

PROVENANCE = Features(
    {
        **{
            key: Value("string")
            for key in ("trace_id", "span_id", "group_id", "model_id", "metadata_json")
        },
        "reward": Value("float64"),
    }
)


DATUM_FEATURES = Features({**PROVENANCE, "datum_json": Value("string")})


def provenance(row: Mapping[str, Any]) -> Provenance:
    """Validate reward, identity, and JSON metadata columns for a dataset row."""
    values: Provenance = {
        "trace_id": row.get("trace_id"),
        "span_id": row.get("span_id"),
        "group_id": row.get("group_id"),
        "model_id": row.get("model_id"),
        "reward": row.get("reward"),
        "metadata_json": row.get("metadata_json") or "{}",
    }
    reward = values["reward"]
    if reward is not None and (type(reward) not in {int, float} or not math.isfinite(reward)):
        raise ValueError("Reward must be finite")
    for key in ("trace_id", "span_id", "group_id", "model_id"):
        if values[key] is not None and not isinstance(values[key], str):
            raise ValueError(f"{key} must be a string")
    metadata = values["metadata_json"] or "{}"
    if not isinstance(json.loads(metadata), dict):
        raise ValueError("metadata_json must contain an object")
    values["metadata_json"] = metadata
    return values
