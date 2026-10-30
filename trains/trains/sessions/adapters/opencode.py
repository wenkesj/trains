# Convert the JSON object produced by `opencode export SESSION_ID`.
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class OpenCodeSessionAdapter:
    """Read OpenCode JSON session exports into chat records."""

    pattern = "*.json"

    def read(self, path: Path) -> dict[str, Any]:
        """Convert an `opencode export` or `opencode session export` artifact."""
        record = self.to_chat(path.read_text())
        record["trace_id"] = record["trace_id"] or path.stem
        return record

    def to_chat(self, session: str | Mapping[str, Any]) -> dict[str, Any]:
        """Convert an exported object or JSON document, including in `Dataset.map`."""
        if isinstance(session, str):
            session = json.loads(session)
        return {
            "messages": _messages_from_export(session),
            "trace_id": session.get("info", {}).get("id"),
        }


def _messages_from_export(session: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Preserve text, reasoning, and completed tool calls in an OpenCode export."""
    messages = []
    for entry in session["messages"]:
        message = {"role": entry["info"]["role"], "content": ""}
        results = []
        for part in entry["parts"]:
            kind = part["type"]
            if kind == "text":
                message["content"] += part["text"]
            elif kind == "reasoning":
                message["reasoning_content"] = message.get("reasoning_content", "") + part["text"]
            elif kind == "tool":
                state = part["state"]
                if state["status"] not in {"completed", "error"}:
                    raise ValueError("Export contains an unfinished OpenCode tool call")
                message.setdefault("tool_calls", []).append(
                    {
                        "id": part["callID"],
                        "type": "function",
                        "function": {"name": part["tool"], "arguments": state["input"]},
                    }
                )
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": part["callID"],
                        "content": state.get("output", state.get("error", "")),
                    }
                )
            elif kind == "file":
                raise ValueError("OpenCode file parts require explicit asset materialization")
        messages.append(message)
        messages.extend(results)
    return messages
