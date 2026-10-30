# Convert Pi JSONL messages as used by Hugging Face agent trace datasets.
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from trains.sessions.adapters.jsonl import read_events, session_events


class PiSessionAdapter:
    """Read pi JSONL session files into chat records."""

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
    """Convert Pi text, thinking, toolCall and toolResult blocks."""
    messages = []
    for event in events:
        if event.get("type") != "message":
            continue
        native = event["message"]
        role = native["role"]
        message = {"role": "tool" if role == "toolResult" else role, "content": ""}
        if role == "toolResult":
            message["tool_call_id"] = native["toolCallId"]
        content = native.get("content", [])
        if isinstance(content, str):
            message["content"] = content
        else:
            for part in content:
                kind = part["type"]
                if kind == "text":
                    message["content"] += part["text"]
                elif kind == "thinking":
                    message["reasoning_content"] = (
                        message.get("reasoning_content", "") + part["thinking"]
                    )
                elif kind == "toolCall":
                    message.setdefault("tool_calls", []).append(
                        {
                            "id": part["id"],
                            "type": "function",
                            "function": {"name": part["name"], "arguments": part["arguments"]},
                        }
                    )
                else:
                    raise ValueError(f"Unsupported Pi content block: {kind}")
        if (
            role == "assistant"
            and native.get("stopReason") in {"error", "aborted"}
            and not any(message.get(key) for key in ("content", "reasoning_content", "tool_calls"))
        ):
            if (native.get("usage") or {}).get("output", 0):
                raise ValueError("Pi error reports generated tokens without captured content")
            continue  # Native request failure placeholder, not a sampled model action.
        messages.append(message)
    return messages
