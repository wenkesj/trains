# Pinned upstream Exercism tasks with separate candidate and verifier workspaces.

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

EXERCISM_REPOSITORY = "https://github.com/exercism/python.git"
EXERCISM_ROOT = Path(__file__).resolve().parents[1] / "tracing" / "exercism"
DEFAULT_TRAIN = ("anagram", "binary-search", "matrix")
DEFAULT_EVAL = ("isogram", "word-count")


@dataclass(frozen=True)
class Exercise:
    """A pinned upstream exercise with separate candidate and verifier files."""

    slug: str
    source: Path
    solutions: tuple[str, ...]
    tests: tuple[str, ...]
    instructions: str
    revision: str
    checksum: str

    @classmethod
    def load(cls, repository: Path, slug: str) -> "Exercise":
        """Read and validate the original exercise files and Git revision."""
        if not re.fullmatch(r"[a-z]+(?:-[a-z]+)*", slug):
            raise ValueError("Exercise names must be upstream slugs")
        revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        source = repository / "exercises" / "practice" / slug
        config = json.loads((source / ".meta" / "config.json").read_text())
        solutions = tuple(config["files"]["solution"])
        tests = tuple(config["files"]["test"])
        files = [*solutions, *tests, ".docs/instructions.md"]
        digest = hashlib.sha256()
        for name in files:
            path = source / name
            if (
                path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(source.resolve())
            ):
                raise ValueError(
                    "Exercise files must be regular files within their source directory"
                )
            digest.update(name.encode() + b"\0" + path.read_bytes())
        return cls(
            slug,
            source,
            solutions,
            tests,
            (source / ".docs/instructions.md").read_text(),
            revision,
            digest.hexdigest(),
        )

    def candidate(self, destination: Path) -> None:
        """Copy starter solutions and instructions into a fresh candidate directory."""
        destination.mkdir(parents=True, exist_ok=False)
        for name in self.solutions:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.source / name, path)
        (destination / "INSTRUCTIONS.md").write_text(self.instructions)

    def verifier(self, candidate: Path, destination: Path) -> None:
        """Build a verifier from original tests and the candidate solution files."""
        destination.mkdir(parents=True, exist_ok=False)
        for name in self.tests:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.source / name, path)
        for name in self.solutions:
            path = candidate / name
            if not path.exists() and not path.is_symlink():
                # A deleted starter is graded by the upstream suite's import error.
                continue
            if (
                not path.is_file()
                or path.is_symlink()
                or not path.resolve().is_relative_to(candidate.resolve())
            ):
                raise ValueError(f"Candidate is missing regular solution file {name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)

    def manifest(self) -> dict:
        """Return the exercise identity, source revision, and file checksum."""
        return {
            "exercise": self.slug,
            "repository": EXERCISM_REPOSITORY,
            "revision": self.revision,
            "checksum": self.checksum,
        }


def validate_split(train: tuple[str, ...], evaluation: tuple[str, ...]) -> None:
    """Reject empty, duplicated, or overlapping training and evaluation exercises."""
    if not train or not evaluation or set(train) & set(evaluation):
        raise ValueError("Training and held-out evaluation exercises must be nonempty and disjoint")
    if len(train) != len(set(train)) or len(evaluation) != len(set(evaluation)):
        raise ValueError("Exercise IDs must be unique within each split")
