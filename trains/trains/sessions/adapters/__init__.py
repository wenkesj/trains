# Adapters for the native session formats supported by Trains.
from trains.sessions.adapters.codex import CodexSessionAdapter
from trains.sessions.adapters.opencode import OpenCodeSessionAdapter
from trains.sessions.adapters.pi import PiSessionAdapter
from trains.sessions.adapters.protocol import SessionAdapter
from trains.sessions.adapters.sts import STSSessionAdapter

__all__ = [
    "CodexSessionAdapter",
    "OpenCodeSessionAdapter",
    "PiSessionAdapter",
    "STSSessionAdapter",
    "SessionAdapter",
]
