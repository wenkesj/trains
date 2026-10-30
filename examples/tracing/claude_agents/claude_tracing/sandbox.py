# Run generated exercise code in disposable Docker containers.

import json
import os
import subprocess
from pathlib import Path
from types import TracebackType
from typing import Self

from recipes.verification import UNITTEST_VERIFIER


class Sandbox:
    """A disposable, network-isolated Docker workspace for running candidate code."""

    def __init__(self, workspace: Path, image: str = "python:3.12-slim") -> None:
        """Bind a candidate directory to an isolated Docker image.

        Args:
            workspace: Candidate directory mounted at /workspace inside the container.
            image: Docker image used to execute candidate code.
        """
        self.workspace, self.image = workspace.resolve(), image
        self.container: str | None = None

    def __enter__(self) -> Self:
        self.container = subprocess.check_output(
            [
                "docker",
                "run",
                "--detach",
                "--init",
                "--network=none",
                "--cpus=2",
                "--memory=1g",
                "--pids-limit=128",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,size=256m",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--mount",
                f"type=bind,src={self.workspace},dst=/workspace",
                "--workdir",
                "/workspace",
                self.image,
                "sleep",
                "infinity",
            ],
            text=True,
        ).strip()
        return self

    def execute(self, command: str, timeout: int = 30) -> str:
        """Run a shell command in the candidate container with a finite timeout."""
        if not self.container:
            raise RuntimeError("Sandbox is not running")
        result = subprocess.run(
            [
                "docker",
                "exec",
                self.container,
                "timeout",
                str(timeout),
                "/bin/sh",
                "-c",
                command,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )
        return json.dumps(
            {
                "exit_code": result.returncode,
                "stdout": result.stdout[-32000:],
                "stderr": result.stderr[-16000:],
            }
        )

    def grade(self) -> dict:
        """Run unchanged upstream tests and return their test pass fraction."""
        if not self.container:
            raise RuntimeError("Sandbox is not running")
        result = subprocess.run(
            ["docker", "exec", self.container, "python", "-B", "-c", UNITTEST_VERIFIER],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode:
            raise RuntimeError(f"Verifier did not complete: {result.stderr[-2000:]}")
        value = json.loads(result.stdout.strip().splitlines()[-1])
        if not value["tests"]:
            raise RuntimeError("Upstream verifier ran no tests")
        return {**value, "stdout": result.stdout, "stderr": result.stderr}

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.container:
            subprocess.run(
                ["docker", "rm", "--force", self.container], check=True, stdout=subprocess.DEVNULL
            )
            self.container = None
