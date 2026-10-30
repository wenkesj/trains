# Cohesion: a module holds one concept and offers a surface a caller can hold in mind. These rules
# report a whole module, so they anchor at line 1 and are exempted per file, with
# `# pygate: noqa: PG401` or with `per-file-ignores`.

from __future__ import annotations

import ast
from dataclasses import dataclass

from pygate.diagnostic import Diagnostic
from pygate.source import SourceModule, assigned_names, exported_names

Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


@dataclass(frozen=True)
class CohesionSettings:
    """Thresholds for the cohesion rules, read from ``[tool.pygate.cohesion]``.

    Attributes:
        min_cluster_size: Smallest group of connected definitions that counts as a concept of its
            own. A smaller one is a stray helper, not a rival concept.
        max_public_names: Widest public surface PG402 allows.
        max_top_level_definitions: Widest module PG403 allows.
    """

    min_cluster_size: int = 2
    max_public_names: int = 12
    max_top_level_definitions: int = 20


def check(module: SourceModule, settings: CohesionSettings) -> list[Diagnostic]:
    """Run PG401 through PG403 over one module.

    Args:
        module: The parsed module.
        settings: Cohesion thresholds.

    Returns:
        Up to one finding per rule, each at line 1.
    """
    tree = module.tree
    found: list[tuple[str, str]] = []
    concepts = [group for group in _clusters(tree) if len(group) >= settings.min_cluster_size]
    if len(concepts) >= 2:
        rendered = "; ".join(f"[{', '.join(group)}]" for group in concepts)
        found.append(
            ("PG401", f"Module holds {len(concepts)} disconnected concepts; split it: {rendered}")
        )
    public = _public_names(tree)
    if len(public) > settings.max_public_names:
        found.append(
            (
                "PG402",
                f"Module offers {len(public)} public names (max {settings.max_public_names}); "
                f"narrow it with __all__ or move a concept out",
            )
        )
    width = len(_definitions(tree))
    if width > settings.max_top_level_definitions:
        found.append(
            (
                "PG403",
                f"Module declares {width} top-level definitions (max "
                f"{settings.max_top_level_definitions})",
            )
        )
    return [Diagnostic(module.path, 1, 1, code, message) for code, message in found]


def _clusters(tree: ast.Module) -> list[list[str]]:
    """Group a module's definitions into connected concepts.

    Args:
        tree: Parsed module.

    Returns:
        Clusters of definition names, largest first, each sorted.

    Note:
        Two definitions join when one references the other, or when both reference the same
        module-level symbol -- shared state and a shared helper are the same evidence of one
        concept, so both edges are drawn the same way.
    """
    definitions = _definitions(tree)
    names = [node.name for node in definitions]
    symbols = _module_symbols(tree)
    defined = set(names)
    parent = {name: name for name in names}

    referrers: dict[str, set[str]] = {}
    for node in definitions:
        for symbol in _referenced_symbols(node, symbols):
            referrers.setdefault(symbol, set()).add(node.name)

    for symbol, users in referrers.items():
        members = sorted(users | ({symbol} if symbol in defined else set()))
        for other in members[1:]:
            _join(parent, members[0], other)

    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(_root(parent, name), []).append(name)
    return sorted((sorted(group) for group in grouped.values()), key=lambda g: (-len(g), g[0]))


def _definitions(tree: ast.Module) -> list[Definition]:
    """Collect the module's top-level functions and classes.

    Args:
        tree: Parsed module.

    Returns:
        Definitions in source order, nested ones excluded.
    """
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]


def _join(parent: dict[str, str], left: str, right: str) -> None:
    """Merge two clusters.

    Args:
        parent: Union-find parent links, mutated in place.
        left: A member of the first cluster.
        right: A member of the second cluster.
    """
    left_root, right_root = _root(parent, left), _root(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _module_symbols(tree: ast.Module) -> set[str]:
    """Collect the symbols the module itself owns.

    Args:
        tree: Parsed module.

    Returns:
        Definition names and module-level assignment targets.

    Note:
        Imported names are deliberately absent. Two definitions that both call ``torch`` share a
        dependency, not a concept, and counting imports would connect every module to itself.
    """
    symbols = {node.name for node in _definitions(tree)}
    for node in tree.body:
        symbols |= assigned_names(node)
    return symbols


def _public_names(tree: ast.Module) -> list[str]:
    """Determine what a module offers its importers.

    Args:
        tree: Parsed module.

    Returns:
        The declared ``__all__`` when there is one, otherwise the non-underscore top-level
        definitions.
    """
    for node in tree.body:
        names = exported_names(node)
        if names is not None:
            return names
    return [node.name for node in _definitions(tree) if not node.name.startswith("_")]


def _referenced_symbols(node: Definition, symbols: set[str]) -> set[str]:
    """Find which of the module's own symbols a definition uses.

    Args:
        node: Top-level definition to inspect, decorators and annotations included.
        symbols: The module's own symbols.

    Returns:
        The referenced symbols, excluding the definition's own name.
    """
    used = {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name) and sub.id in symbols}
    used.discard(node.name)
    return used


def _root(parent: dict[str, str], name: str) -> str:
    """Find the representative of a name's cluster.

    Args:
        parent: Union-find parent links, path-compressed in place.
        name: Name to resolve.

    Returns:
        The cluster representative.
    """
    while parent[name] != name:
        parent[name] = parent[parent[name]]
        name = parent[name]
    return name
