# Which files a check covers. A directory is listed through git when it is inside a repository, so
# whatever .gitignore hides -- virtualenvs, node_modules, build output, data trees -- is never even
# walked, which is what ruff does too; each file's configuration then excludes what it names.

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from pygate.settings import Configurations


def python_files(paths: Sequence[Path], configurations: Configurations) -> list[Path]:
    """Collect the Python files a check covers.

    Args:
        paths: Files and directories named by the caller.
        configurations: Settings per file, for exclusion.

    Returns:
        Sorted files. A file named explicitly is still subject to its configuration's excludes.

    Raises:
        FileNotFoundError: When a named path does not exist.
    """
    found: set[Path] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path}: no such file or directory")
        candidates = [path] if path.is_file() else _walk(path)
        found.update(
            candidate
            for candidate in candidates
            if not configurations.for_path(candidate).is_excluded(candidate)
        )
    return sorted(found)


def _walk(directory: Path) -> list[Path]:
    """List a directory's Python files, through git when possible.

    Args:
        directory: Directory to list.

    Returns:
        Its Python files, at any depth.
    """
    listed = _git_listed(directory)
    return listed if listed is not None else sorted(directory.rglob("*.py"))


def _git_listed(directory: Path) -> list[Path] | None:
    """Ask git for the tracked and unignored Python files beneath a directory.

    Args:
        directory: Directory to list.

    Returns:
        The files that exist on disk, or ``None`` when the directory is not in a repository or git
        is unavailable.
    """
    argv = ["git", "-C", str(directory), "ls-files", "--cached", "--others", "--exclude-standard"]
    try:
        done = subprocess.run(
            [*argv, "-z", "--", "*.py"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None
    if done.returncode != 0:
        return None
    # A tracked file deleted in the working tree is still listed by --cached.
    return [directory / name for name in done.stdout.split("\0") if (directory / name).is_file()]
