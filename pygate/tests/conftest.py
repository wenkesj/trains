from __future__ import annotations

import ast
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

from pygate.source import SourceModule


@pytest.fixture
def parse() -> Callable[[str], SourceModule]:
    """Build a module from an indented source literal, its first line being line 1."""

    def build(text: str) -> SourceModule:
        source = textwrap.dedent(text).lstrip("\n")
        return SourceModule(Path("example.py"), source, ast.parse(source))

    return build


@pytest.fixture
def project(tmp_path: Path) -> Callable[..., Path]:
    """Write a pyproject.toml and one module into a temporary project, returning the module."""

    def write(source: str, config: str = "[tool.pygate]\n", name: str = "module.py") -> Path:
        (tmp_path / "pyproject.toml").write_text(textwrap.dedent(config).lstrip("\n"))
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(source).lstrip("\n"))
        return target

    return write
