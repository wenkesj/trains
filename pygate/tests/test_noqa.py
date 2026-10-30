from __future__ import annotations

from pathlib import Path

from pygate.check import check_paths
from pygate.settings import Overrides

MUTATION = """
_CACHE = {{}}
def remember(key):
    _CACHE[key] = 1{comment}
"""

FOUR_FUNCTIONS = """
{directive}
def a(): return b()
def b(): return 1
def c(): return d()
def d(): return 2
"""


def run(target: Path) -> list[tuple[str, int]]:
    result = check_paths([target], Overrides(), target.parent)
    return [(d.code, d.line) for d in result.diagnostics]


def test_a_line_directive_suppresses_its_code_on_its_line(project):
    assert run(project(MUTATION.format(comment="  # noqa: PG202"))) == []


def test_codes_are_matched_regardless_of_case_or_spacing(project):
    assert run(project(MUTATION.format(comment="  # NOQA:pg202"))) == []


def test_a_directive_for_another_pygate_code_leaves_the_finding_and_is_itself_unused(project):
    assert run(project(MUTATION.format(comment="  # noqa: PG201"))) == [
        ("PG202", 3),
        ("PG001", 3),
    ]


def test_a_blanket_directive_suppresses_and_is_left_to_ruff(project):
    assert run(project(MUTATION.format(comment="  # noqa"))) == []


def test_ruff_codes_in_the_same_directive_are_not_ours_to_report(project):
    assert run(project("VALUE = 1  # noqa: E501, PGH003\n")) == []


def test_a_file_directive_suppresses_a_module_finding(project):
    source = FOUR_FUNCTIONS.format(directive="# pygate: noqa: PG401")
    assert run(project(source)) == []


def test_an_unused_file_directive_is_reported(project):
    assert run(project("# pygate: noqa: PG401\nVALUE = 1\n")) == [("PG001", 1)]


def test_a_directive_for_a_rule_that_is_not_enabled_suppresses_nothing(project):
    target = project(
        MUTATION.format(comment="  # noqa: PG202"), config='[tool.pygate]\nignore = ["PG202"]\n'
    )
    assert run(target) == [("PG001", 3)]


def test_directive_text_inside_a_string_is_not_a_directive(project):
    assert run(project('TEXT = "# noqa: PG202"\n')) == []
