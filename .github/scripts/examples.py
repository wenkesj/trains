"""Discover isolated examples and check their Python sources."""

import argparse
import configparser
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IGNORED = {".git", ".venv", "__pycache__", "node_modules", "var", "artifacts", "dist"}


def _walk(root):
    modules = configparser.ConfigParser()
    modules.read(ROOT / ".gitmodules")
    vendors = {
        (ROOT / modules[section]["path"]).resolve() for section in modules.sections()
    }
    for directory, children, names in os.walk(root):
        children[:] = sorted(
            name
            for name in children
            if name not in IGNORED and (Path(directory) / name).resolve() not in vendors
        )
        yield Path(directory), sorted(names)


def _projects():
    return sorted(
        directory.relative_to(ROOT).as_posix()
        for directory, names in _walk(ROOT / "examples")
        if "pyproject.toml" in names
    )


def _sources(project):
    files = [
        (directory / name).relative_to(ROOT).as_posix()
        for directory, names in _walk(ROOT / project)
        for name in names
        if name.endswith((".py", ".pyi"))
    ]
    if not files:
        raise ValueError(f"No Python sources found in {project}")
    return files


def main():
    """Emit the CI matrix or check examples using their own dependency environments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["matrix", "lint", "type", "test", "check"])
    parser.add_argument("example", nargs="?")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    projects = _projects()
    if args.command == "matrix":
        matrix = json.dumps({"example": projects}, separators=(",", ":"))
        if args.github_output:
            with args.github_output.open("a") as output:
                output.write(f"matrix={matrix}\n")
        print(matrix)
        return 0
    if args.example and args.example not in projects:
        parser.error("EXAMPLE must be one of: " + ", ".join(projects))
    status = 0
    for project in [args.example] if args.example else projects:
        files = _sources(project)
        if args.command in {"lint", "check"}:
            print(f"Ruff: {project}", flush=True)
            status |= subprocess.run(
                [str(ROOT / ".venv/bin/ruff"), "check", *files], cwd=ROOT, check=False
            ).returncode
            print(f"Pygate: {project}", flush=True)
            status |= subprocess.run(
                [str(ROOT / ".venv/bin/pygate"), "check", *files], cwd=ROOT, check=False
            ).returncode
        if args.command in {"type", "test", "check"}:
            python = ROOT / project / ".venv"
            if not (python / "bin/python").exists():
                print(
                    f"Missing environment: run uv sync --project {project} --locked --all-extras",
                    file=sys.stderr,
                )
                status = 1
                continue
            if args.command == "test":
                tests = ROOT / project / "tests"
                if not tests.is_dir():
                    print(f"No unit test suite: {project}", flush=True)
                    continue
                print(f"pytest: {project}", flush=True)
                status |= subprocess.run(
                    [
                        str(python / "bin/python"),
                        "-m",
                        "pytest",
                        "-c",
                        str(ROOT / project / "pyproject.toml"),
                        str(tests),
                        "-q",
                    ],
                    cwd=ROOT,
                    check=False,
                ).returncode
                continue
            print(f"ty: {project}", flush=True)
            status |= subprocess.run(
                [
                    str(ROOT / ".venv/bin/ty"),
                    "check",
                    "--project",
                    project,
                    "--python",
                    str(python),
                    *files,
                ],
                cwd=ROOT,
                check=False,
            ).returncode
    return status


if __name__ == "__main__":
    raise SystemExit(main())
