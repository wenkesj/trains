# One parsed module, and the readings of its syntax tree that more than one rule makes.

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class SourceModule:
    """A module read and parsed once, then handed to every rule.

    Attributes:
        path: The file, as the caller named it.
        text: Its source.
        tree: The parsed module.
    """

    path: Path
    text: str
    tree: ast.Module


def read_module(path: Path) -> SourceModule:
    """Read and parse one file.

    Args:
        path: File to read.

    Returns:
        The parsed module.

    Raises:
        SyntaxError: When the file does not parse.
        UnicodeDecodeError: When the file is not UTF-8.
    """
    text = path.read_text(encoding="utf-8")
    return SourceModule(path=path, text=text, tree=ast.parse(text, filename=str(path)))


def assigned_names(node: ast.stmt) -> set[str]:
    """Read the plain names one assignment binds, unpacking included.

    Args:
        node: Candidate statement.

    Returns:
        The bound names, empty when the statement binds nothing by plain name.
    """
    if isinstance(node, ast.AnnAssign | ast.AugAssign):
        targets: list[ast.expr] = [node.target]
    elif isinstance(node, ast.Assign):
        targets = list(node.targets)
    else:
        return set()
    names: set[str] = set()
    for target in targets:
        elements = target.elts if isinstance(target, ast.Tuple | ast.List) else [target]
        names.update(element.id for element in elements if isinstance(element, ast.Name))
    return names


def exported_names(node: ast.stmt) -> list[str] | None:
    """Read the string literals assigned to ``__all__``.

    Args:
        node: Candidate module-level statement.

    Returns:
        The exported names, or ``None`` when the statement is not an ``__all__`` assignment of a
        plain literal sequence.
    """
    targets = getattr(node, "targets", [])
    if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
        return None
    value = getattr(node, "value", None)
    if not isinstance(value, ast.List | ast.Tuple):
        return None
    names = [
        element.value
        for element in value.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]
    return names if len(names) == len(value.elts) else None
