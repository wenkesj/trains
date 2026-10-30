from typing import TypedDict

from trains.formats.records import Message, ToolDefinition


class Provenance(TypedDict):
    """Optional native identities, task groups, rewards, and serialized metadata."""

    trace_id: str | None
    span_id: str | None
    group_id: str | None
    model_id: str | None
    reward: float | None
    metadata_json: str


class ChatRecord(TypedDict, total=False):
    """Input provenance is optional; chat validation requires messages."""

    messages: list[Message]
    tools: list[ToolDefinition] | None
    chat_template_kwargs: dict[str, object] | None
    trace_id: str | None
    span_id: str | None
    group_id: str | None
    model_id: str | None
    reward: float | None
    metadata_json: str


class ChatRow(Provenance):
    """Normalized Arrow rows retain messages, rendering settings, and provenance."""

    messages: list[Message]
    tools: list[ToolDefinition] | None
    chat_template_kwargs: dict[str, object] | None


class DatumRow(Provenance):
    """A serialized native training datum and its provenance columns."""

    datum_json: str
