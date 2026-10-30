from __future__ import annotations

import subprocess

from pygate.files import python_files
from pygate.settings import Configurations, Overrides


def test_a_directory_outside_git_is_walked_less_its_excludes(project):
    target = project("VALUE = 1\n")
    root = target.parent
    for relative in ("pkg/inner.py", ".venv/lib/site.py", "notes.txt"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    files = python_files([root], Configurations(Overrides(), root))
    assert files == [root / "module.py", root / "pkg" / "inner.py"]


def test_a_directory_inside_git_skips_what_gitignore_hides(project):
    target = project("VALUE = 1\n")
    root = target.parent
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / ".gitignore").write_text("data/\n")
    (root / "data").mkdir()
    (root / "data" / "huge.py").write_text("")
    files = python_files([root], Configurations(Overrides(), root))
    assert files == [root / "module.py"]
