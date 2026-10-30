# Convert Codex rollout response items, ignoring duplicated UI events.
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from trains.sessions.adapters.jsonl import read_events, session_events


class CodexSessionAdapter:
    """Read codex JSONL session files into chat records."""

    pattern = "*.jsonl"

    def read(self, path: Path) -> dict[str, Any]:
        """Convert one session, preserving its native session identity."""
        record = self.to_chat(read_events(path))
        record["trace_id"] = record["trace_id"] or path.stem
        return record

    def to_chat(self, session: str | Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Convert a raw JSONL session or decoded events, including in `Dataset.map`."""
        events = session_events(session)
        header = next((event for event in events if event.get("type") == "session_meta"), {})
        identity = header.get("payload", header).get("id")
        return {"messages": _messages_from_events(events), "trace_id": identity}


def _messages_from_events(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract model conversation items from a Codex session JSONL stream."""
    messages: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") != "response_item":
            continue
        item = event["payload"]
        kind = item.get("type")
        if kind == "message":
            parts = item.get("content", [])
            content = (
                "".join(p["text"] for p in parts)
                if all(p.get("type") in {"input_text", "output_text", "text"} for p in parts)
                else parts
            )
            messages.append({"role": item["role"], "content": content})
        elif kind in {"function_call", "custom_tool_call"}:
            arguments = item.get("arguments", {})
            if kind == "custom_tool_call":
                arguments = {"input": item["input"]}
            call = {
                "id": item["call_id"],
                "type": "function",
                "function": {"name": item["name"], "arguments": arguments},
            }
            # Responses emits parallel calls as separate items before any results.
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "tool_calls": [call]})
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            messages.append(
                {"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]}
            )
    return messages
