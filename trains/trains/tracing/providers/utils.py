from collections.abc import Iterable, Iterator
from typing import Any

from trains.formats.tracing import TraceRecord
from trains.tracing.adapters.span import decoded, decoded_attribute


def traces_from_spans(spans: Iterable[dict]) -> Iterator[TraceRecord]:
    """Group OpenInference span exports by their native trace identity."""
    groups = {}
    for value in spans:
        span = _span_row(value)
        identifier = span.get("TraceId")
        if not identifier:
            raise ValueError("Every exported span must retain its native trace ID")
        groups.setdefault(str(identifier), []).append(span)
    for identifier, values in groups.items():
        if not any(span["SpanAttributes"].get("openinference.span.kind") for span in values):
            raise ValueError("Provider export must retain OpenInference span attributes")
        ids = [s["SpanId"] for s in values]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate span IDs in provider export")
        yield TraceRecord(id=identifier, spans=values)


def frame_records(frame: Any) -> list[dict]:
    """Iterate provider exports as mappings without assuming one dataframe SDK."""
    missing = [name for name in frame.index.names if name is not None and name not in frame.columns]
    frame = frame.reset_index(level=missing) if missing else frame.reset_index(drop=True)
    rows = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    return [{key: value for key, value in row.items() if value is not None} for row in rows]


def _span_row(value: dict) -> dict:
    """Normalize one provider-owned span model into portable attributes."""
    if "SpanAttributes" in value:
        return {
            **value,
            "SpanAttributes": _flatten_attributes(value["SpanAttributes"]),
        }
    context = value.get("context") or {}
    attrs = {
        **(value.get("attributes") or {}),
        **{
            k.removeprefix("attributes."): v
            for k, v in value.items()
            if k.startswith("attributes.")
        },
    }
    return {
        "TraceId": value.get("context.trace_id", context.get("trace_id", value.get("trace_id"))),
        "SpanId": value.get(
            "context.span_id", context.get("span_id", value.get("id", value.get("span_id", "")))
        ),
        "ParentSpanId": value.get("parent_id", value.get("parent_span_id")),
        "status_code": value.get("status_code"),
        "SpanName": value.get("name", ""),
        "SpanAttributes": _flatten_attributes(attrs),
        "events": value.get("events", []),
        "native": value,
    }


def _flatten_attributes(attributes: dict, prefix: str = "") -> dict:
    """Normalize native nested exports while preserving semantic JSON payload leaves."""
    attributes = decoded(attributes)
    if not isinstance(attributes, dict):
        raise ValueError("OpenInference attributes must be an object")
    leaves = {
        "input.value",
        "output.value",
        "llm.invocation_parameters",
        "tool.parameters",
        "tool.json_schema",
    }
    result = {}
    for name, value in attributes.items():
        key = f"{prefix}.{name}" if prefix else name
        if key in leaves or key.endswith((".tool.json_schema", ".tool_call.function.arguments")):
            result[key] = decoded_attribute(key, value)
        elif isinstance(value, dict):
            result.update(_flatten_attributes(value, key))
        elif isinstance(value, list) and (
            key in {"llm.input_messages", "llm.output_messages", "llm.tools"}
            or key.endswith((".message.tool_calls", ".message.contents"))
        ):
            result.update(
                _flatten_attributes({str(index): item for index, item in enumerate(value)}, key)
            )
        else:
            result[key] = value
    return result
