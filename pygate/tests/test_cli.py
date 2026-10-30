from __future__ import annotations

import json

from pygate.cli import main


def test_a_clean_file_passes(project, monkeypatch, capsys):
    target = project("VALUE = 1\n")
    monkeypatch.chdir(target.parent)
    assert main(["check", "module.py"]) == 0
    assert capsys.readouterr().out.strip() == "All checks passed!"


def test_findings_print_in_ruffs_concise_format_and_exit_one(project, monkeypatch, capsys):
    target = project('"""Docstring."""\n')
    monkeypatch.chdir(target.parent)
    assert main(["check", "module.py"]) == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("module.py:1:1: PG102 ")
    assert lines[-1] == "Found 1 error."


def test_exit_zero_and_json_output(project, monkeypatch, capsys):
    target = project('"""Docstring."""\n')
    monkeypatch.chdir(target.parent)
    assert main(["check", "--exit-zero", "--output-format", "json", "module.py"]) == 0
    [finding] = json.loads(capsys.readouterr().out)
    assert finding["code"] == "PG102"
    assert finding["location"] == {"row": 1, "column": 1}


def test_a_command_line_ignore_drops_a_rule(project, monkeypatch):
    target = project('"""Docstring."""\n')
    monkeypatch.chdir(target.parent)
    assert main(["check", "--ignore", "PG102", "module.py"]) == 0


OUT_OF_ZONE = "def _helper():\n    return 1\n\n\nclass Public:\n    pass\n"


def test_diff_previews_the_reorder_without_writing(project, monkeypatch, capsys):
    target = project(OUT_OF_ZONE)
    monkeypatch.chdir(target.parent)
    assert main(["check", "--diff", "module.py"]) == 1
    assert capsys.readouterr().out.startswith("--- a/module.py\n+++ b/module.py\n")
    assert target.read_text() == OUT_OF_ZONE


def test_fix_leaves_a_file_without_a_zone_finding_alone(project, monkeypatch):
    source = "import os\n\n\ndef public():\n    return os.sep\n\n\nalias = int\n"
    target = project(source)
    monkeypatch.chdir(target.parent)
    assert main(["check", "--diff", "module.py"]) == 0
    assert target.read_text() == source


def test_fix_writes_the_reorder_and_reports_what_remains(project, monkeypatch, capsys):
    target = project(OUT_OF_ZONE)
    monkeypatch.chdir(target.parent)
    assert main(["check", "--fix", "module.py"]) == 0
    assert target.read_text().index("class Public") < target.read_text().index("def _helper")
    assert "All checks passed!" in capsys.readouterr().out


def test_rule_explains_one_code_and_lists_them_all(capsys):
    assert main(["rule", "pg205"]) == 0
    assert "mutable-singleton" in capsys.readouterr().out
    assert main(["rule", "--all"]) == 0
    assert "PG403" in capsys.readouterr().out
    assert main(["rule", "PG999"]) == 2


def test_an_unreadable_path_or_configuration_exits_two(project, monkeypatch):
    target = project("VALUE = 1\n", config="[tool.pygate]\nselct = []\n")
    monkeypatch.chdir(target.parent)
    assert main(["check", "module.py"]) == 2
    assert main(["check", "missing.py"]) == 2
