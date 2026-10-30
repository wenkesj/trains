# The interface needed to import a native session format.
from pathlib import Path
from typing import Any, Protocol


class SessionAdapter[SessionT](Protocol):
    """Read one native session artifact into a chat record."""

    pattern: str

    def to_chat(self, session: SessionT) -> dict[str, Any]:
        """Convert an in-memory native session to a chat record for HF mapping."""
        ...

    def read(self, path: Path) -> dict[str, Any]:
        """Return messages and optional chat provenance from one session file."""
        ...
