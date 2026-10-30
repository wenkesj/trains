# Shared JSONL framing for native session adapters.
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read JSON objects, reporting the line number of malformed native events."""
    events = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError as error:
                raise ValueError(f"Line {number}: invalid JSON") from error
            if not isinstance(event, dict):
                raise ValueError(f"Line {number}: expected an event object")
            events.append(event)
    return events


def session_events(value: str | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Accept an in-memory JSONL document or its decoded event objects."""
    if isinstance(value, str):
        events = [json.loads(line) for line in value.splitlines() if line.strip()]
    else:
        events = list(value)
    if any(not isinstance(event, Mapping) for event in events):
        raise ValueError("Session events must be JSON objects")
    return [dict(event) for event in events]
