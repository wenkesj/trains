# The two records every other module speaks: a rule the checker knows, and a finding one made.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class Diagnostic:
    """One finding at one place.

    Attributes:
        path: File the finding is in, as the caller named it.
        line: 1-based line the finding anchors to. A ``# noqa`` for it sits on this line.
        column: 1-based column.
        code: Rule code, such as ``PG101``.
        message: What is wrong, in one sentence.
    """

    path: Path
    line: int
    column: int
    code: str
    message: str


@dataclass(frozen=True)
class Rule:
    """A rule the checker knows.

    Attributes:
        code: Stable identity, used by ``select``, ``ignore`` and ``noqa``.
        name: Kebab-case name, for people.
        summary: One line saying what the rule reports.
        explanation: Why it is a defect, and the shape that fixes it.
        default: Whether the rule is selected when the configuration names no ``select``.
    """

    code: str
    name: str
    summary: str
    explanation: str
    default: bool = True
