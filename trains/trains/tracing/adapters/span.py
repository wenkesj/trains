import json
from typing import Any

from pydantic import BaseModel, Field


class Span(BaseModel):
    """A normalized native span identity and its model-call attributes."""

    span_id: str = ""
    name: str = ""
    attributes: dict = Field(default_factory=dict)
    events: list[dict] = Field(default_factory=list)

    @classmethod
    def from_row(cls, row: dict) -> "Span":
        """Normalize one exported provider row into a span."""
        attrs = {
            key: decoded_attribute(key, value)
            for key, value in row.get("SpanAttributes", {}).items()
        }
        events = row.get("events", [])
        if not events:
            events = [
                {"name": n, "attributes": {k: decoded_attribute(k, v) for k, v in a.items()}}
                for n, a in zip(
                    row.get("Events.Name", []), row.get("Events.Attributes", []), strict=True
                )
            ]
        return cls(
            span_id=row.get("SpanId", ""),
            name=row.get("SpanName", ""),
            attributes=attrs,
            events=events,
        )


def decoded(value: Any) -> Any:
    """Decode JSON strings while retaining already structured native values."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def decoded_attribute(name: str, value: Any) -> Any:
    """Decode structured attributes without interpreting captured message text as JSON."""
    if name in {"input.value", "output.value"}:
        result = decoded(value)
        return result if isinstance(result, (dict, list)) else value
    if name in {"llm.invocation_parameters", "metadata"} or name.endswith(
        (".tool.json_schema", ".tool_call.function.arguments")
    ):
        return decoded(value)
    return value
