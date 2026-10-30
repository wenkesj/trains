# The proof between a rewrite and the disk. The reorder moves raw text, so it could in principle drop
# a decorator, swallow a comment or truncate a body; three independent comparisons have to agree it
# did none of that before anything is written.

from __future__ import annotations

import ast
from collections import Counter

from pygate.fix.chunks import chunks


def proof_failures(before: str, after: str) -> tuple[str, ...]:
    """Prove a rewrite permuted top-level statements and changed nothing else.

    Args:
        before: The module as it was.
        after: The module as it would be written.

    Returns:
        Every way the rewrite failed to be a pure reordering; empty when it is one.
    """
    try:
        after_tree = ast.parse(after)
    except SyntaxError as exc:
        return (f"the rewrite does not parse: {exc}",)
    before_tree = ast.parse(before)
    failures: list[str] = []
    if sorted(ast.dump(n) for n in before_tree.body) != sorted(
        ast.dump(n) for n in after_tree.body
    ):
        failures.append("the top-level statements are not the same set, so this is not a reorder")
    before_cut = chunks(before, before_tree)
    after_cut = chunks(after, after_tree)
    if Counter(c.label for c in before_cut) != Counter(c.label for c in after_cut):
        failures.append("the set of definitions changed")
    if sorted(c.text for c in before_cut) != sorted(c.text for c in after_cut):
        failures.append(
            "a definition's own source text changed: a decorator or a leading comment did not "
            "travel with the definition it belongs to"
        )
    before_lines = Counter(line.strip() for line in before.splitlines() if line.strip())
    after_lines = Counter(line.strip() for line in after.splitlines() if line.strip())
    if before_lines != after_lines:
        lost = sorted((before_lines - after_lines).elements())[:3]
        gained = sorted((after_lines - before_lines).elements())[:3]
        failures.append(f"lines were lost or gained. Lost {lost}, gained {gained}")
    return tuple(failures)
