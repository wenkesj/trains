# One file's zone fix: the reorder, the proof, and the diff a caller writes or prints.

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass

from pygate.fix.proof import proof_failures
from pygate.fix.reorder import rebuild


@dataclass(frozen=True)
class Fix:
    """What the zone fix made of one module.

    Attributes:
        text: The source to write, or the original when nothing moved or the proof failed.
        moved: How many top-level statements changed position.
        diff: Unified diff from the original to ``text``; empty when nothing moved.
        refusals: Why a region, or the whole file, was left alone.
    """

    text: str
    moved: int
    diff: str
    refusals: tuple[str, ...]


def fix_source(text: str, tree: ast.Module, where: str) -> Fix:
    """Reorder one module into zone order, proving the rewrite before offering it.

    Args:
        text: Module source.
        tree: The same source, parsed.
        where: Path for messages and diff headers.

    Returns:
        The fix. A rewrite whose proof fails is withheld and its reasons reported.
    """
    after, refusals, moved = rebuild(text, tree, where)
    if after == text:
        return Fix(text, 0, "", refusals)
    broken = proof_failures(text, after)
    if broken:
        return Fix(text, 0, "", (*refusals, *(f"{where}: {reason}" for reason in broken)))
    diff = "".join(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{where}",
            tofile=f"b/{where}",
        )
    )
    return Fix(after, moved, diff, refusals)
