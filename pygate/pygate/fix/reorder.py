# The zone reorder: setup assignments rise beside the imports, then each stretch of governed
# definitions is sorted into zone order without breaking an import-time dependency. It never
# crosses an import or a statement that runs, and it refuses a stretch a loose comment sits in.

from __future__ import annotations

import ast
import heapq

from pygate.fix.chunks import Chunk, Definition, assignment_targets, chunks
from pygate.rules.layout import definition_zone


def rebuild(source: str, tree: ast.Module, where: str) -> tuple[str, tuple[str, ...], int]:
    """Rewrite a module with its setup beside its imports and its definitions in zone order.

    Args:
        source: Module text.
        tree: The same text, parsed.
        where: Module path, for the messages.

    Returns:
        The rewritten text, the reasons any region was left alone, and how many chunks moved.
    """
    cut = chunks(source, tree)
    if not cut:
        return source, (), 0
    lines = source.splitlines(keepends=True)
    gap = {
        cut[i].index: "".join(lines[cut[i].end : cut[i + 1].start - 1]) for i in range(len(cut) - 1)
    }
    welded = {index for index, text in gap.items() if "#" in text}
    placed = _hoisted(cut, tree, welded)
    refusals: list[str] = []
    for start, stop in _regions(placed):
        loose = next(
            (i for i in range(start, stop - 1) if "#" in gap.get(placed[i].index, "")), None
        )
        if loose is not None:
            refusals.append(
                f"{where}: a comment between {placed[loose].label} and {placed[loose + 1].label} "
                f"belongs to neither. Attach it to one of them by hand, then rerun"
            )
            continue
        arranged = _ordered(placed[start:stop])
        if arranged is None:
            refusals.append(
                f"{where}: {', '.join(c.label for c in placed[start:stop])} form an import-time "
                f"dependency cycle, so no order satisfies them"
            )
            continue
        placed[start:stop] = arranged
    # A pair the reorder never separated keeps its author's spacing; a pair it invented gets the
    # spacing its two members call for.
    body = "".join(
        chunk.text
        + (
            ""
            if i == len(placed) - 1
            else gap[chunk.index]
            if placed[i + 1].index == chunk.index + 1
            else _separator(chunk, placed[i + 1])
        )
        for i, chunk in enumerate(placed)
    )
    head = "".join(lines[: cut[0].start - 1])
    tail = "".join(lines[cut[-1].end :])
    moved = sum(1 for was, now in zip(cut, placed, strict=True) if was.index != now.index)
    return head + body + tail, tuple(refusals), moved


def _hoisted(cut: list[Chunk], tree: ast.Module, welded: set[int]) -> list[Chunk]:
    """Lift the module's setup assignments out from between its definitions.

    Args:
        cut: The module's chunks, in source order.
        tree: The parsed module the chunks came from.
        welded: Original indices whose following gap holds a comment, which nothing may cross.

    Returns:
        The chunks with every hoistable setup assignment moved up beside the imports. One is
        hoistable only when it names nothing the module defines and crosses nothing that runs.
    """
    settled = list(cut)
    for _ in range(len(cut)):
        first = next((i for i, c in enumerate(settled) if c.zone is not None), None)
        if first is None:
            return settled
        movable = next(
            (
                i
                for i in range(first + 1, len(settled))
                if _is_setup(tree.body[settled[i].index])
                and not _is_attribute_docstring(tree.body, settled[i].index + 1)
                and welded.isdisjoint({settled[i].index, settled[i].index - 1})
                and all(_is_inert(tree.body[c.index]) for c in settled[first:i])
                and settled[i].reads.isdisjoint(
                    {name for c in settled[first:i] for name in c.binds}
                )
            ),
            None,
        )
        if movable is None:
            return settled
        settled.insert(first, settled.pop(movable))
    return settled


def _is_attribute_docstring(body: list[ast.stmt], index: int) -> bool:
    """Decide whether the statement at ``index`` documents the assignment above it.

    Args:
        body: The module's top-level statements.
        index: Position to inspect.

    Returns:
        Whether that position holds a bare string literal.
    """
    if index >= len(body):
        return False
    node = body[index]
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _is_inert(node: ast.stmt) -> bool:
    """Decide whether a statement can be stepped over without changing what the module does.

    Args:
        node: Module-level statement.

    Returns:
        Whether the statement only binds a name, rather than running something observable.
    """
    return isinstance(node, ast.Assign | ast.AnnAssign | Definition)


def _is_setup(node: ast.stmt) -> bool:
    """Decide whether a statement is module setup that may be lifted beside the imports.

    Args:
        node: Module-level statement.

    Returns:
        Whether it is a plain assignment binding names the zone order does not govern.
    """
    return (
        isinstance(node, ast.Assign | ast.AnnAssign)
        and definition_zone(node) is None
        and bool(assignment_targets(node))
    )


def _ordered(members: list[Chunk]) -> list[Chunk] | None:
    """Sort one region into zone order without breaking its import-time dependencies.

    Args:
        members: The region's chunks, in source order.

    Returns:
        The zone-earliest ordering that respects every dependency edge, or ``None`` when a cycle
        leaves no valid order.
    """
    owner = {name: position for position, chunk in enumerate(members) for name in chunk.binds}
    blockers = [
        {owner[name] for name in chunk.reads if name in owner and owner[name] != position}
        for position, chunk in enumerate(members)
    ]
    dependents: list[list[int]] = [[] for _ in members]
    for position, needs in enumerate(blockers):
        for earlier in needs:
            dependents[earlier].append(position)
    outstanding = [len(needs) for needs in blockers]
    ready = [(members[p].zone, p) for p, count in enumerate(outstanding) if count == 0]
    heapq.heapify(ready)
    result: list[Chunk] = []
    while ready:
        _, position = heapq.heappop(ready)
        result.append(members[position])
        for dependent in dependents[position]:
            outstanding[dependent] -= 1
            if outstanding[dependent] == 0:
                heapq.heappush(ready, (members[dependent].zone, dependent))
    return result if len(result) == len(members) else None


def _regions(cut: list[Chunk]) -> list[tuple[int, int]]:
    """Find the stretches of a module a reorder is allowed to touch.

    Args:
        cut: The module's chunks.

    Returns:
        Half-open index ranges of two or more consecutive governed statements. Imports,
        ``if TYPE_CHECKING`` blocks and executable setup are barriers the reorder never crosses.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    sentinel = Chunk("", "", None, frozenset(), frozenset(), 0, 0, -1)
    for index, chunk in enumerate([*cut, sentinel]):
        if chunk.zone is None:
            if start is not None and index - start > 1:
                spans.append((start, index))
            start = None
        elif start is None:
            start = index
    return spans


def _separator(above: Chunk, below: Chunk) -> str:
    """Choose the blank lines between two definitions a reorder has just paired.

    Args:
        above: The chunk that now comes first.
        below: The chunk that now follows it.

    Returns:
        One blank line between two constants, two otherwise.
    """
    return "\n" if above.zone == 0 and below.zone == 0 else "\n\n"
