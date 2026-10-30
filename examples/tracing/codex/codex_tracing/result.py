# Explicit terminal conditions for a finite-horizon coding attempt.

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AgentResult:
    """The final assistant response and the reason an attempt stopped."""

    response: str | None
    stop_reason: Literal["completed", "max_turns", "context_limit"]
