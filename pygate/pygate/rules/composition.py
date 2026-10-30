# Composition: how much one class may hold -- the slots it assigns on `self` and the methods it
# defines. Past either line a class is several collaborators sharing a constructor, and neither
# cohesion rule can see it, because one class is one concept and one public name.

from __future__ import annotations

import ast
from dataclasses import dataclass

from pygate.diagnostic import Diagnostic
from pygate.source import FunctionNode, SourceModule


@dataclass(frozen=True)
class CompositionSettings:
    """Thresholds for the composition rules, read from ``[tool.pygate.composition]``.

    Attributes:
        max_instance_attributes: Distinct ``self.<name>`` slots PG301 allows one class.
        max_methods: Methods PG302 allows one class to define in its own body.
    """

    max_instance_attributes: int = 12
    max_methods: int = 12


def check(module: SourceModule, settings: CompositionSettings) -> list[Diagnostic]:
    """Run PG301 and PG302 over every class in one module, nested classes included.

    Args:
        module: The parsed module.
        settings: Composition thresholds.

    Returns:
        One diagnostic per class past each limit, at the class statement.
    """
    found: list[Diagnostic] = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.ClassDef):
            found.extend(_class_findings(module, node, settings))
    return found


def _instance_attributes(node: ast.ClassDef) -> set[str]:
    """Find the slots a class assigns on itself.

    Args:
        node: Class definition to inspect.

    Returns:
        Every distinct name assigned as ``self.<name>``.

    Note:
        Declared fields are deliberately absent. A dataclass states its record in annotations that
        are typed, ordered and visible at a glance, which is the shape this rule pushes toward;
        what it counts is state assembled imperatively, one ``self.x =`` at a time.
    """
    return {
        sub.attr
        for sub in ast.walk(node)
        if isinstance(sub, ast.Attribute)
        and isinstance(sub.ctx, ast.Store)
        and isinstance(sub.value, ast.Name)
        and sub.value.id == "self"
    }


def _methods(node: ast.ClassDef) -> list[FunctionNode]:
    """Collect the methods a class defines in its own body.

    Args:
        node: Class definition to inspect.

    Returns:
        Its direct function and coroutine definitions, in source order.
    """
    return [
        child for child in node.body if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def _class_findings(
    module: SourceModule, node: ast.ClassDef, settings: CompositionSettings
) -> list[Diagnostic]:
    """Measure one class against both limits.

    Args:
        module: The module the class is in.
        node: The class.
        settings: Composition thresholds.

    Returns:
        Zero, one or two findings.
    """
    slots = len(_instance_attributes(node))
    defined = len(_methods(node))
    found: list[Diagnostic] = []
    if slots > settings.max_instance_attributes:
        found.append(
            Diagnostic(
                module.path,
                node.lineno,
                node.col_offset + 1,
                "PG301",
                f"`{node.name}` assigns {slots} slots on self (max "
                f"{settings.max_instance_attributes}); group the slots that change together into "
                f"an object it holds",
            )
        )
    if defined > settings.max_methods:
        found.append(
            Diagnostic(
                module.path,
                node.lineno,
                node.col_offset + 1,
                "PG302",
                f"`{node.name}` defines {defined} methods (max {settings.max_methods}); move the "
                f"methods that share slots into a collaborator",
            )
        )
    return found
