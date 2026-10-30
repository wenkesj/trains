# Layout: how long a function body may run, what a module may open with, where `__all__` sits and
# how it is ordered, the order definitions are stated in, and which functions may nest.
# `definition_zone` and `zone_offenders` are shared with the fixer so both use the same ordering.
# Nested-definition detection is internal to this rule module.

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from pygate.diagnostic import Diagnostic
from pygate.source import FunctionNode, SourceModule, exported_names

Finding = tuple[str, ast.stmt, str]


#: Zone order. A module reads top-down as its own table of contents: what it configures, what it
#: defines, what it offers, what it hides, and what it exports. The index is the order.
ZONE_NAMES: dict[int, str] = {
    0: "module constants",
    1: "public classes",
    2: "private classes",
    3: "public functions",
    4: "private functions",
    5: "__all__",
}


@dataclass(frozen=True)
class LayoutSettings:
    """Thresholds for the layout rules, read from ``[tool.pygate.layout]``.

    Attributes:
        max_function_body_lines: Longest body PG101 allows, in source lines, docstring excluded.
    """

    max_function_body_lines: int = 50


def check(module: SourceModule, settings: LayoutSettings) -> list[Diagnostic]:
    """Run PG101 through PG106 over one module.

    Args:
        module: The parsed module.
        settings: Layout thresholds.

    Returns:
        One diagnostic per finding.
    """
    tree = module.tree
    findings: list[Finding] = [
        *_long_bodies(tree, settings.max_function_body_lines),
        *_module_docstring(tree),
        *_dunder_all(tree),
        *(("PG105", node, message) for node, message in zone_offenders(tree)),
        *(("PG106", node, message) for node, message in _nested_offenders(tree)),
    ]
    return [
        Diagnostic(module.path, node.lineno, node.col_offset + 1, code, message)
        for code, node, message in findings
    ]


def definition_zone(node: ast.stmt) -> int | None:
    """Place one top-level statement in the zone its kind belongs to.

    Args:
        node: Module-level statement.

    Returns:
        The zone index, or ``None`` for the statements this order does not govern -- imports,
        executable setup, and lowercase module-level assignment. Those stay where they are.
    """
    if isinstance(node, ast.ClassDef):
        return 1 if not node.name.startswith("_") else 2
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return 3 if not node.name.startswith("_") else 4
    names = _assignment_targets(node)
    if names is None:
        return None
    if names == ["__all__"]:
        return 5
    return 0 if all(_is_constant_name(name) for name in names) else None


def zone_offenders(tree: ast.Module) -> list[tuple[ast.stmt, str]]:
    """Find the definitions a module states out of zone order.

    Args:
        tree: Parsed module.

    Returns:
        Each offending statement, with a message naming the zone it sits in and the one it
        belongs to.
    """
    defined_above = {
        node.name: node.lineno
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }
    offenders: list[tuple[ast.stmt, str]] = []
    deepest = -1
    for node in tree.body:
        zone = definition_zone(node)
        if zone is None:
            continue
        if zone < deepest and not (zone == 0 and _constant_needs_its_place(node, defined_above)):
            offenders.append(
                (
                    node,
                    f"`{_statement_label(node)}` sits in the {ZONE_NAMES[deepest]} zone; it "
                    f"belongs in {ZONE_NAMES[zone]}",
                )
            )
        deepest = max(deepest, zone)
    return offenders


def _nested_offenders(tree: ast.Module) -> list[tuple[ast.stmt, str]]:
    """Find the nested functions that are module-level helpers in hiding.

    Args:
        tree: Parsed module.

    Returns:
        Each inner function that captures nothing and is not structurally nested, with a message.
    """
    offenders: list[tuple[ast.stmt, str]] = []

    def walk(node: ast.AST, enclosing: list[FunctionNode]) -> None:
        for child in ast.iter_child_nodes(node):
            inner = isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            if inner and enclosing:
                _record_if_free_standing(child, enclosing, offenders)  # type: ignore[arg-type]
            walk(child, [*enclosing, child] if inner else enclosing)  # type: ignore[list-item]

    walk(tree, [])
    return offenders


def _assignment_targets(node: ast.stmt) -> list[str] | None:
    """Read the plain module-level names one assignment binds.

    Args:
        node: Candidate module-level statement.

    Returns:
        The bound names, or ``None`` when the statement is not an assignment to plain names.
    """
    if isinstance(node, ast.AnnAssign | ast.AugAssign):
        targets: list[ast.expr] = [node.target]
    elif isinstance(node, ast.Assign):
        targets = list(node.targets)
    else:
        return None
    names = [target.id for target in targets if isinstance(target, ast.Name)]
    return names if names and len(names) == len(targets) else None


def _body_line_span(node: FunctionNode) -> int:
    """Measure a function body in source lines, ignoring its docstring.

    Args:
        node: Function or coroutine definition to measure.

    Returns:
        Count of source lines spanned by the body statements.
    """
    body = node.body
    first = body[0]
    is_docstring = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    statements = body[1:] if is_docstring and len(body) > 1 else body
    if not statements:
        return 0
    start = statements[0].lineno
    end = max(statement.end_lineno or statement.lineno for statement in statements)
    return end - start + 1


def _bound_names(node: FunctionNode) -> set[str]:
    """Collect every name a function binds anywhere inside itself.

    Args:
        node: Function or coroutine definition.

    Returns:
        Parameters, assignment targets, imports, caught exceptions, and nested definitions.
    """
    args = node.args
    slots = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    names = {slot.arg for slot in slots if slot is not None}
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
            names.add(sub.id)
        elif isinstance(sub, ast.arg):
            names.add(sub.arg)
        elif isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(sub.name)
        elif isinstance(sub, ast.Import | ast.ImportFrom):
            names.update((alias.asname or alias.name).split(".")[0] for alias in sub.names)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            names.add(sub.name)
    return names


def _constant_needs_its_place(node: ast.stmt, defined_above: dict[str, int]) -> bool:
    """Decide whether a constant is held down by a definition it is built from.

    Args:
        node: Module-level statement, already known to sit in the constant zone.
        defined_above: Line number of every class and function the module defines.

    Returns:
        Whether the constant reads a name defined above it.

    Note:
        A registry, a singleton or a tuple of instances is a constant whose value is the
        definitions above it. Hoisting it into the constant zone is not a reordering, it is a
        ``NameError``, so the zone rule exempts exactly these. The exemption is a property of the
        statement: it stops applying the moment the constant stops naming what precedes it.
    """
    own = set(_assignment_targets(node) or [])
    used = {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)} & set(defined_above)
    return any(defined_above[name] < node.lineno for name in used - own)


def _dunder_all(tree: ast.Module) -> list[Finding]:
    """Find an ``__all__`` that is not last, or not sorted.

    Args:
        tree: Parsed module.

    Returns:
        PG103 and PG104 findings.
    """
    findings: list[Finding] = []
    last = len(tree.body) - 1
    for index, node in enumerate(tree.body):
        targets = getattr(node, "targets", [])
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        if index != last:
            findings.append(
                (
                    "PG103",
                    node,
                    f"`__all__` is statement {index + 1} of {last + 1}; it belongs at the bottom",
                )
            )
        names = exported_names(node)
        if names is not None and names != sorted(names, key=_export_sort_key):
            findings.append(
                (
                    "PG104",
                    node,
                    "`__all__` is unsorted: constants, then classes, then the rest, in natural "
                    "order within each",
                )
            )
    return findings


def _export_sort_key(name: str) -> tuple[int, tuple[int | str, ...]]:
    """Build the isort-style ordering key for one exported name.

    Args:
        name: Exported symbol.

    Returns:
        A category rank paired with a natural-order key.

    Note:
        Casing sets the category -- SCREAMING_SNAKE_CASE constants first, then CamelCase classes,
        then everything else -- and a natural sort orders each category so ``lane2`` precedes
        ``lane10``.
    """
    stripped = name.strip("_")
    if stripped.isupper():
        category = 0
    elif stripped[:1].isupper():
        category = 1
    else:
        category = 2
    natural = tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))
    return category, natural


def _free_names(node: FunctionNode) -> set[str]:
    """Collect the names a function reads without binding them itself.

    Args:
        node: Function or coroutine definition.

    Returns:
        The names it must find in an enclosing scope, in a builtin, or at module level.
    """
    reads = {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
    }
    return reads - _bound_names(node)


def _is_constant_name(name: str) -> bool:
    """Decide whether a bound name reads as a module constant.

    Args:
        name: Bound name.

    Returns:
        Whether the name is SCREAMING_SNAKE_CASE, leading underscores aside.
    """
    return name.strip("_").isupper() and any(char.isalpha() for char in name)


def _is_handed_out(parent: FunctionNode, name: str) -> bool:
    """Decide whether a parent returns the inner function rather than calling it.

    Args:
        parent: The enclosing function.
        name: The inner function's name.

    Returns:
        Whether some ``return`` yields the function object itself.

    Note:
        A name in call position is a use, not a hand-off. ``return helper(x)`` returns a value and
        leaves ``helper`` hidden; ``return helper`` is a factory, which is structural.
    """
    for statement in ast.walk(parent):
        if not isinstance(statement, ast.Return) or statement.value is None:
            continue
        called = {call.func for call in ast.walk(statement.value) if isinstance(call, ast.Call)}
        for sub in ast.walk(statement.value):
            if isinstance(sub, ast.Name) and sub.id == name and sub not in called:
                return True
    return False


def _long_bodies(tree: ast.Module, limit: int) -> list[Finding]:
    """Find function bodies longer than the limit.

    Args:
        tree: Parsed module.
        limit: Longest allowed body, in source lines.

    Returns:
        PG101 findings, nested functions and methods included.
    """
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        span = _body_line_span(node)
        if span > limit:
            findings.append(
                (
                    "PG101",
                    node,
                    f"`{node.name}` has a {span}-line body (max {limit}); split it into named "
                    f"helpers",
                )
            )
    return findings


def _module_docstring(tree: ast.Module) -> list[Finding]:
    """Find a module-level docstring.

    Args:
        tree: Parsed module.

    Returns:
        A PG102 finding, or nothing.
    """
    if ast.get_docstring(tree) is None:
        return []
    return [
        (
            "PG102",
            tree.body[0],
            "Module-level docstring; state the contract in code and comments instead",
        )
    ]


def _record_if_free_standing(
    node: FunctionNode, enclosing: list[FunctionNode], offenders: list[tuple[ast.stmt, str]]
) -> None:
    """Record an inner function that has no reason to be nested.

    Args:
        node: The inner function.
        enclosing: The chain of functions it sits inside, innermost last.
        offenders: Findings, appended to in place.
    """
    if node.decorator_list or _is_handed_out(enclosing[-1], node.name):
        return
    visible = set().union(*(_bound_names(scope) for scope in enclosing)) - {node.name}
    if _free_names(node) & visible:
        return
    offenders.append(
        (
            node,
            f"`{node.name}()` inside `{enclosing[-1].name}()` captures nothing from it; lift it "
            f"to module level",
        )
    )


def _statement_label(node: ast.stmt) -> str:
    """Name a top-level statement for a message.

    Args:
        node: Module-level statement.

    Returns:
        The definition's name, or the names an assignment binds.
    """
    name = getattr(node, "name", None)
    return str(name) if name else ", ".join(_assignment_targets(node) or ["<statement>"])
