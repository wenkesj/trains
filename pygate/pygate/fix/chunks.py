# A module cut into movable pieces: one chunk per top-level statement, carrying the comments and
# decorators that must travel with it and the names it binds and needs. The zone fix reorders
# chunks and nothing smaller, which is what keeps a move from separating a definition from its
# documentation.

from __future__ import annotations

import ast
from dataclasses import dataclass

from pygate.rules.layout import definition_zone

Definition = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class Chunk:
    """One top-level statement together with everything that must travel with it.

    Attributes:
        label: The definition's name, or the names an assignment binds.
        text: Exact source of the leading comments, the decorators and the statement.
        zone: Zone index, or ``None`` for a statement the zone order does not govern.
        binds: Names the statement binds at module level.
        reads: Names the statement needs bound before it can itself be executed.
        start: First source line, 1-based and inclusive.
        end: Last source line, 1-based and inclusive.
        index: Position in the original module body.
    """

    label: str
    text: str
    zone: int | None
    binds: frozenset[str]
    reads: frozenset[str]
    start: int
    end: int
    index: int


def assignment_targets(node: ast.stmt) -> list[str]:
    """Read the plain module-level names one assignment binds.

    Args:
        node: Candidate module-level statement.

    Returns:
        The bound names, empty when the statement is not an assignment to plain names.
    """
    if isinstance(node, ast.AnnAssign | ast.AugAssign):
        targets: list[ast.expr] = [node.target]
    elif isinstance(node, ast.Assign):
        targets = list(node.targets)
    else:
        return []
    return [target.id for target in targets if isinstance(target, ast.Name)]


def chunks(source: str, tree: ast.Module) -> list[Chunk]:
    """Cut a module into one chunk per top-level statement.

    Args:
        source: Module text.
        tree: The same text, parsed.

    Returns:
        One chunk per statement, in source order.

    Note:
        A chunk reaches back over the comment block written directly above its statement. That
        block is the statement's documentation; leaving it behind while the definition moves would
        silently reattach it to whatever landed in its place.
    """
    lines = source.splitlines(keepends=True)
    annotations_run = _annotations_evaluated(tree)
    result: list[Chunk] = []
    floor = 1
    for index, node in enumerate(tree.body):
        start, end = _statement_span(node)
        while start > floor and lines[start - 2].lstrip().startswith("#"):
            start -= 1
        binds = [node.name] if isinstance(node, Definition) else assignment_targets(node)
        result.append(
            Chunk(
                label=_statement_label(node),
                text="".join(lines[start - 1 : end]),
                zone=definition_zone(node),
                binds=frozenset(binds),
                reads=_import_time_reads(node, annotations_run),
                start=start,
                end=end,
                index=index,
            )
        )
        floor = end + 1
    return result


def _annotations_evaluated(tree: ast.Module) -> bool:
    """Decide whether a module's annotations run when its definitions are reached.

    Args:
        tree: Parsed module.

    Returns:
        Whether annotations are evaluated rather than stored as strings.
    """
    return not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _definition_time_names(node: ast.stmt, annotations_run: bool) -> set[str]:
    """Collect the names a definition reads while it is being defined.

    Args:
        node: A class or function definition.
        annotations_run: Whether the module evaluates its annotations.

    Returns:
        Names read by decorators, bases, parameter defaults, class bodies and, when the module does
        not defer them, annotations.
    """
    names: set[str] = set()
    parts: list[ast.AST] = list(getattr(node, "decorator_list", []))
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        parts += [*node.args.defaults, *[d for d in node.args.kw_defaults if d]]
        if annotations_run:
            slots = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            parts += [a.annotation for a in slots if a.annotation is not None]
            parts += [node.returns] if node.returns is not None else []
    elif isinstance(node, ast.ClassDef):
        parts += [*node.bases, *[k.value for k in node.keywords]]
        for statement in node.body:
            if isinstance(statement, Definition):
                names |= _definition_time_names(statement, annotations_run)
            elif isinstance(statement, ast.AnnAssign) and not annotations_run:
                parts += [statement.value] if statement.value is not None else []
            else:
                parts.append(statement)
    for part in parts:
        names |= _loads(part)
    return names


def _import_time_reads(node: ast.stmt, annotations_run: bool) -> frozenset[str]:
    """Collect the names a statement needs bound before it can execute.

    Args:
        node: Module-level statement.
        annotations_run: Whether the module evaluates its annotations.

    Returns:
        Every name read while the statement itself runs. Function bodies run later and constrain
        nothing; decorators, bases, defaults and class bodies run at definition time.
    """
    if isinstance(node, Definition):
        return frozenset(_definition_time_names(node, annotations_run))
    value = getattr(node, "value", None)
    return frozenset() if value is None else frozenset(_loads(value))


def _loads(node: ast.AST) -> set[str]:
    """Collect the names a fragment of code reads, never the names it stores.

    Args:
        node: Any expression or statement.

    Returns:
        The names loaded.
    """
    return {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
    }


def _statement_label(node: ast.stmt) -> str:
    """Name a top-level statement for reports and for the proof.

    Args:
        node: Module-level statement.

    Returns:
        The definition's name, the names an assignment binds, or the statement's kind.
    """
    name = getattr(node, "name", None)
    if name:
        return str(name)
    bound = assignment_targets(node)
    return ", ".join(bound) if bound else f"<{type(node).__name__}>"


def _statement_span(node: ast.stmt) -> tuple[int, int]:
    """Measure the lines a statement occupies, its decorators included.

    Args:
        node: Module-level statement.

    Returns:
        First and last source line, both 1-based and inclusive.
    """
    start = node.lineno
    for decorator in getattr(node, "decorator_list", []):
        start = min(start, decorator.lineno)
    return start, node.end_lineno or node.lineno
