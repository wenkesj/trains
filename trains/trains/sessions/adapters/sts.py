# Hugging Face Session Trace Simple Format conversion.
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from trains.sessions.adapters.jsonl import read_events, session_events


class STSSessionAdapter:
    """Read sts JSONL session files into chat records."""

    pattern = "*.jsonl"

    def read(self, path: Path) -> dict[str, Any]:
        """Convert one session, preserving its native session identity."""
        record = self.to_chat(read_events(path))
        record["trace_id"] = record["trace_id"] or path.stem
        return record

    def to_chat(self, session: str | Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Convert a raw JSONL session or decoded events, including in `Dataset.map`."""
        events = session_events(session)
        header = next((event for event in events if event.get("type") == "session"), {})
        identity = header.get("payload", header).get("id")
        return {"messages": _messages_from_events(events), "trace_id": identity}


def _messages_from_events(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract STS envelopes, translating their camel-case message fields."""
    aliases = {
        "reasoningContent": "reasoning_content",
        "toolCalls": "tool_calls",
        "toolCallId": "tool_call_id",
    }
    return [
        {aliases.get(key, key): value for key, value in event["message"].items()}
        for event in events
        if event.get("type") == "message"
    ]
