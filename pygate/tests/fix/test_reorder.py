"""The zone fix moves whole definitions and nothing else.

It rewrites raw text, so the interesting question is not whether it sorts correctly but whether a
move can lose something on the way. These exercise the proof that stands between a rewrite and the
disk: a dropped decorator, a stranded comment and a truncated body each have to be caught, and a
statement that cannot move safely has to stay where it is.
"""

from __future__ import annotations

import ast

import pytest

from pygate.fix.apply import fix_source
from pygate.fix.proof import proof_failures
from pygate.fix.reorder import rebuild

DECORATED = '''from __future__ import annotations

import functools


def _helper() -> int:
    return 1


#: What the cache is for.
@functools.cache
def value() -> int:
    """Documented."""
    return _helper()


CONSTANT = 2
'''


def _rebuild(source: str) -> str:
    text, refusals, _moved = rebuild(source, ast.parse(source), "example.py")
    assert not refusals, refusals
    return text


def test_a_move_carries_the_decorator_and_the_comment_above_it() -> None:
    rewritten = _rebuild(DECORATED)

    assert not proof_failures(DECORATED, rewritten)
    assert rewritten.index("CONSTANT = 2") < rewritten.index("def value()")
    assert "#: What the cache is for.\n@functools.cache\ndef value()" in rewritten


def test_the_rewrite_is_idempotent() -> None:
    once = _rebuild(DECORATED)
    assert _rebuild(once) == once


@pytest.mark.parametrize(
    ("damage", "replacement"),
    [
        ("@functools.cache\n", ""),
        ("#: What the cache is for.\n", ""),
        ('    """Documented."""\n', ""),
        ("CONSTANT = 2", "CONSTANT = 3"),
    ],
)
def test_the_proof_refuses_a_rewrite_that_is_not_a_reordering(
    damage: str, replacement: str
) -> None:
    damaged = _rebuild(DECORATED).replace(damage, replacement)

    assert proof_failures(DECORATED, damaged)


def test_a_comment_between_two_definitions_stops_the_reorder() -> None:
    source = (
        "from __future__ import annotations\n\n\n"
        "def _private() -> int:\n    return 1\n\n\n"
        "# A banner belonging to neither.\n\n\n"
        "def public() -> int:\n    return 2\n"
    )

    text, refusals, moved = rebuild(source, ast.parse(source), "example.py")

    assert text == source
    assert moved == 0
    assert refusals and "belongs to neither" in refusals[0]


def test_setup_rises_to_the_imports_but_never_crosses_something_that_runs() -> None:
    liftable = (
        "from __future__ import annotations\n\n"
        "import typer\n\n\n"
        "def public() -> int:\n    return 1\n\n\n"
        "app = typer.Typer()\n\n\n"
        "def _private() -> int:\n    return 2\n"
    )
    pinned = liftable.replace("import typer\n", "import typer\n\ntyper.main.get_command\n")

    lifted = _rebuild(liftable)
    held = _rebuild(pinned)

    assert lifted.index("app = typer.Typer()") < lifted.index("def public()")
    assert held.index("app = typer.Typer()") > held.index("typer.main.get_command")


def test_a_constant_built_from_a_definition_is_not_hoisted_into_a_name_error() -> None:
    source = (
        "from __future__ import annotations\n\n\n"
        "def alpha() -> int:\n    return 1\n\n\n"
        "REGISTRY = (alpha,)\n\n\n"
        "class Late:\n    pass\n"
    )

    rewritten = _rebuild(source)

    assert not proof_failures(source, rewritten)
    assert rewritten.index("def alpha()") < rewritten.index("REGISTRY = (alpha,)")
    assert rewritten.index("class Late") < rewritten.index("def alpha()")


def test_fix_source_offers_a_proved_rewrite_with_its_diff() -> None:
    fixed = fix_source(DECORATED, ast.parse(DECORATED), "example.py")

    assert fixed.moved and not fixed.refusals
    assert fixed.diff.startswith("--- a/example.py")
    assert fix_source(fixed.text, ast.parse(fixed.text), "example.py").moved == 0
