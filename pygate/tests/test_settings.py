from __future__ import annotations

from pathlib import Path

import pytest

from pygate.settings import ConfigError, Configurations, Overrides


def settings_for(target: Path, overrides: Overrides | None = None):
    return Configurations(overrides or Overrides(), target.parent).for_path(target)


def test_the_default_selection_is_every_rule_but_the_wide_module_smell(project):
    selected = settings_for(project("VALUE = 1\n")).selected
    assert "PG101" in selected and "PG205" in selected
    assert "PG403" not in selected


def test_select_takes_prefixes_and_ignore_and_extend_select_adjust_it(project):
    config = """
    [tool.pygate]
    select = ["PG2"]
    extend-select = ["PG403"]
    ignore = ["PG205"]
    """
    selected = settings_for(project("VALUE = 1\n", config=config)).selected
    assert selected == {"PG201", "PG202", "PG203", "PG204", "PG403"}


def test_a_command_line_select_replaces_the_configured_one(project):
    target = project("VALUE = 1\n", config='[tool.pygate]\nselect = ["PG1"]\n')
    assert settings_for(target, Overrides(select=("PG301",))).selected == {"PG301"}


def test_per_file_ignores_match_globs_relative_to_the_config(project):
    target = project("", config='[tool.pygate.per-file-ignores]\n"**/__init__.py" = ["PG4"]\n')
    package_init = target.parent / "pkg" / "__init__.py"
    package_init.parent.mkdir()
    package_init.write_text("")
    settings = settings_for(target)
    assert "PG401" not in settings.enabled_for(package_init)
    assert "PG401" in settings.enabled_for(target)


def test_a_bare_exclude_matches_at_any_depth_and_a_slashed_one_is_anchored(project):
    config = '[tool.pygate]\nextend-exclude = ["generated", "pkg/only"]\n'
    target = project("", config=config)
    settings = settings_for(target)
    root = target.parent
    assert settings.is_excluded(root / "generated" / "a.py")
    assert settings.is_excluded(root / "pkg" / "generated" / "a.py")
    assert settings.is_excluded(root / "pkg" / "only" / "a.py")
    assert not settings.is_excluded(root / "other" / "pkg" / "only" / "a.py")
    assert settings.is_excluded(root / "node_modules" / "a.py")


def test_the_nearest_configuration_governs_a_file(project):
    target = project("", config='[tool.pygate]\nselect = ["PG1"]\n')
    nested = target.parent / "sub" / "module.py"
    nested.parent.mkdir()
    (nested.parent / "pyproject.toml").write_text('[tool.pygate]\nselect = ["PG3"]\n')
    nested.write_text("")
    assert settings_for(nested).selected == {"PG301", "PG302"}


def test_thresholds_are_read_per_family(project):
    config = "[tool.pygate.composition]\nmax-methods = 3\n"
    assert settings_for(project("", config=config)).composition.max_methods == 3


@pytest.mark.parametrize(
    "config",
    [
        "[tool.pygate]\nselct = []\n",
        '[tool.pygate]\nselect = ["PG9"]\n',
        "[tool.pygate]\nselect = \"PG1\"\n",
        "[tool.pygate.layout]\nmax-function-body-lines = 0\n",
        "[tool.pygate.layout]\nmax-lines = 10\n",
    ],
)
def test_a_configuration_that_cannot_be_honoured_is_refused(project, config):
    with pytest.raises(ConfigError):
        settings_for(project("", config=config))
