import base64
import dataclasses
import json
from typing import Any


def datum_mapping(value: Any) -> dict[str, Any]:
    """Convert a native SDK datum into a JSON-compatible mapping."""
    value = json.loads(json.dumps(value, default=_native_json, allow_nan=False))
    return {key: item for key, item in value.items() if item is not None}


def _native_json(value: Any) -> Any:
    """Serialize a native datum without changing provider field names."""
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if all(hasattr(value, key) for key in ("data", "dtype", "shape")):
        return {
            "data": value.data,
            "dtype": value.dtype,
            **({"shape": value.shape} if value.shape is not None else {}),
        }
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python", exclude_none=True)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
            if getattr(value, field.name) is not None
        }
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Unsupported native datum value: {type(value).__name__}")
