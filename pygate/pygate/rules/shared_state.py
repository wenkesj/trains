# Process-wide mutable state: a module-level name something writes after import, in the five
# spellings that state takes. A read-only table is deliberately absent. `HANDLERS = {"a": handle_a}`
# is a constant whose value happens to be a dict; what makes a name global is that something
# writes it.

from __future__ import annotations

import ast

from pygate.diagnostic import Diagnostic
from pygate.source import FunctionNode, SourceModule, assigned_names

#: Methods that write through a reference. Calling one on a module-level name mutates state every
#: importer shares, whoever holds the reference.
MUTATING_METHODS = frozenset(
    {
        "add",
        "append",
        "appendleft",
        "clear",
        "difference_update",
        "discard",
        "extend",
        "extendleft",
        "insert",
        "intersection_update",
        "move_to_end",
        "pop",
        "popitem",
        "put",
        "remove",
        "reverse",
        "setdefault",
        "sort",
        "symmetric_difference_update",
        "update",
    }
)

#: Constructors whose only purpose is to guard mutable state. A module-level lock is evidence of a
#: module-level thing to lock, so it is named in its own right.
SYNCHRONISATION_CONSTRUCTORS = frozenset(
    {"BoundedSemaphore", "Condition", "Event", "Lock", "RLock", "Semaphore", "local"}
)

#: Decorators that install a process-wide memo table keyed by a function's arguments.
MEMOISING_DECORATORS = frozenset({"cache", "cached_property", "lru_cache"})

#: Why each spelling is a defect, by the rule that names it.
REASONS: dict[str, str] = {
    "PG201": "is rebound at runtime by a `global` statement",
    "PG202": "is module-level state a function mutates, shared by every importer",
    "PG203": "guards module-level state, so that state is shared too",
    "PG204": "installs an unbounded process-wide memo table",
    "PG205": "holds one module-level instance whose fields are written after construction",
}

Located = ast.stmt | ast.expr


def check(module: SourceModule) -> list[Diagnostic]:
    """Run PG201 through PG205 over one module.

    Args:
        module: The parsed module.

    Returns:
        One diagnostic per offending name. A name that is state in several spellings at once is
        reported once, under the first rule in code order that sees it, at the place that rule
        points to -- which is where its ``# noqa`` goes.
    """
    tree = module.tree
    bound = _module_bindings(tree)
    spellings: tuple[tuple[str, dict[str, Located]], ...] = (
        ("PG201", _global_offences(tree)),
        ("PG202", _mutation_offences(tree, bound)),
        ("PG203", _primitive_offences(tree)),
        ("PG204", _memo_offences(tree)),
        ("PG205", _singleton_offences(tree)),
    )
    first: dict[str, tuple[str, Located]] = {}
    for code, found in spellings:
        for name, node in found.items():
            first.setdefault(name, (code, node))
    return [
        Diagnostic(module.path, node.lineno, node.col_offset + 1, code, f"`{name}` {REASONS[code]}")
        for name, (code, node) in first.items()
    ]


def _call_name(node: ast.Call) -> str:
    """Name the callable one call invokes.

    Args:
        node: Call expression.

    Returns:
        The bare attribute or identifier, empty when it is neither.
    """
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return node.func.id if isinstance(node.func, ast.Name) else ""


def _constructed_class(node: ast.Call) -> str:
    """Find the class at the root of a constructor or named factory call.

    Args:
        node: Call expression.

    Returns:
        The identifier the call chain starts from, empty when it starts elsewhere.
    """
    target: ast.expr = node.func
    while isinstance(target, ast.Attribute):
        target = target.value
    return target.id if isinstance(target, ast.Name) else ""


def _decorator_name(node: ast.expr) -> str:
    """Name the decorator an expression applies, called or not.

    Args:
        node: Decorator expression.

    Returns:
        The bare attribute or identifier, empty when it is neither.
    """
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Attribute):
        return target.attr
    return target.id if isinstance(target, ast.Name) else ""


def _global_offences(tree: ast.Module) -> dict[str, Located]:
    """Find the names some function rebinds with ``global``.

    Args:
        tree: Parsed module.

    Returns:
        Each declared name mapped to the statement that declares it.
    """
    found: dict[str, Located] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            for name in node.names:
                found.setdefault(name, node)
    return found


def _memo_offences(tree: ast.Module) -> dict[str, Located]:
    """Find the functions carrying a process-wide memo table.

    Args:
        tree: Parsed module.

    Returns:
        Each memoised function name mapped to its definition.
    """
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(_decorator_name(d) in MEMOISING_DECORATORS for d in node.decorator_list)
    }


def _module_bindings(tree: ast.Module) -> set[str]:
    """Collect the names a module binds at module level.

    Args:
        tree: Parsed module.

    Returns:
        Every name assigned by a top-level statement, ``__all__`` aside.
    """
    names: set[str] = set()
    for node in tree.body:
        names |= assigned_names(node)
    names.discard("__all__")
    return names


def _mutation_offences(tree: ast.Module, bound: set[str]) -> dict[str, Located]:
    """Find the module-level names a function writes through without rebinding.

    Args:
        tree: Parsed module.
        bound: The names the module binds at module level.

    Returns:
        Each mutated name mapped to the first write.

    Note:
        Only writes inside a function count. A module that fills a table between its own
        top-level statements is still building a constant; one that fills it from a function is
        carrying state between calls.
    """
    found: dict[str, Located] = {}
    for scope in ast.walk(tree):
        if not isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for name, node in _writes_within(scope, bound).items():
            found.setdefault(name, node)
    return found


def _primitive_offences(tree: ast.Module) -> dict[str, Located]:
    """Find the module-level synchronisation primitives.

    Args:
        tree: Parsed module.

    Returns:
        Each guarding name mapped to the statement that constructs it.
    """
    found: dict[str, Located] = {}
    for node in tree.body:
        value = getattr(node, "value", None)
        if isinstance(value, ast.Call) and _call_name(value) in SYNCHRONISATION_CONSTRUCTORS:
            for name in assigned_names(node):
                found.setdefault(name, node)
    return found


def _root_name(node: ast.expr) -> str:
    """Find the identifier at the base of a reference chain.

    Args:
        node: Expression to walk down, through subscripts and attributes.

    Returns:
        The base identifier, empty when the chain does not start at a plain name.
    """
    while isinstance(node, ast.Subscript | ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def _singleton_offences(tree: ast.Module) -> dict[str, Located]:
    """Find the module-level instances that carry state between calls.

    Args:
        tree: Parsed module.

    Returns:
        Each singleton name mapped to the statement that constructs it.

    Note:
        This is the rule's own escape hatch, closed: moving a bare ``_CACHE = {}`` onto a field
        of a module-level object changes how the state is spelled and not who shares it.
    """
    stateful = _stateful_classes(tree)
    found: dict[str, Located] = {}
    for node in tree.body:
        value = getattr(node, "value", None)
        if isinstance(value, ast.Call) and _constructed_class(value) in stateful:
            for name in assigned_names(node):
                found.setdefault(name, node)
    return found


def _stateful_classes(tree: ast.Module) -> set[str]:
    """Find the classes that keep writing themselves after construction.

    Args:
        tree: Parsed module.

    Returns:
        The names of classes that assign ``self.<field>`` from a method other than ``__init__``.
    """
    stateful: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for method in node.body:
            if not isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if method.name != "__init__" and any(_writes_self(sub) for sub in ast.walk(method)):
                stateful.add(node.name)
                break
    return stateful


def _writes_self(node: ast.AST) -> bool:
    """Decide whether one node assigns a field on ``self``.

    Args:
        node: Node to test.

    Returns:
        Whether it stores to ``self.<field>`` or calls a mutating method on one.
    """
    if isinstance(node, ast.Subscript | ast.Attribute) and isinstance(node.ctx, ast.Store):
        return _root_name(node) == "self"
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in MUTATING_METHODS
        and _root_name(node.func.value) == "self"
    )


def _writes_within(scope: FunctionNode, bound: set[str]) -> dict[str, Located]:
    """Find the writes one function makes to module-level names.

    Args:
        scope: Function or coroutine definition to inspect.
        bound: The names the module binds at module level.

    Returns:
        Each written module-level name mapped to the node that writes it.
    """
    found: dict[str, Located] = {}

    def record(target: ast.expr, node: Located) -> None:
        name = _root_name(target)
        if name in bound:
            found.setdefault(name, node)

    for node in ast.walk(scope):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store | ast.Del):
            record(node, node)
        elif isinstance(node, ast.AugAssign):
            record(node.target, node)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                record(target, node)
        elif (
            isinstance(node, ast.Call)
            and _call_name(node) in MUTATING_METHODS
            and isinstance(node.func, ast.Attribute)
        ):
            record(node.func.value, node)
    return found
