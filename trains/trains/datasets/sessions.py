# Build chat datasets from persisted coding-agent sessions.
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from trains.datasets.chat import chat_dataset_from_records, chat_row
from trains.sessions.adapters.protocol import SessionAdapter


def chat_dataset_from_sessions(path: str | Path, *, adapter: SessionAdapter[Any]) -> Dataset:
    """Read one session or recursively import matching files from a directory.

    Args:
        path: A native session artifact or a directory containing session files.
        adapter: A native session reader, such as `CodexSessionAdapter()`.

    Returns:
        One validated conversation per file, with native IDs and source provenance.

    Raises:
        ValueError: No matching files exist, or an artifact cannot form a chat record.
    """
    root = Path(path).expanduser()
    paths = sorted(root.rglob(adapter.pattern)) if root.is_dir() else [root]
    if not paths:
        raise ValueError(f"No sessions matching {adapter.pattern} found in {root}")
    rows = []
    for source in paths:
        try:
            record = adapter.read(source)
            metadata = json.loads(record.get("metadata_json") or "{}")
            record["metadata_json"] = json.dumps({**metadata, "source": str(source)})
            rows.append(chat_row(record))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Cannot import session {source}: {error}") from error
    return chat_dataset_from_records(rows)


__all__ = ["chat_dataset_from_sessions"]
