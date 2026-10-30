# Run real coding CLIs and their native integrations in disposable profiles.

import asyncio
import json
import os
import subprocess
from functools import partial
from uuid import uuid4

from opencode_tracing.model_proxy import ModelProxy
from opencode_tracing.result import AgentResult
from opencode_tracing.sandbox import Sandbox
from opencode_tracing.training import Policy


async def solve(sandbox: Sandbox, policy: Policy) -> AgentResult:
    """Run the native coding harness against the candidate workspace and policy."""
    name = "trains-tracing-" + uuid4().hex
    with ModelProxy(policy) as proxy:
        environment = {
            "TRAINS_MODEL_URL": proxy.url,
            "TINKER_API_KEY": os.environ["TINKER_API_KEY"],
        }
        environment["TRACE_TO_LANGSMITH"] = "true"
        keys = ("LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT")
        environment.update({key: os.environ[key] for key in keys if key in os.environ})
        command = _harness_command(name, sandbox.workspace, environment)
        try:
            result = await asyncio.to_thread(
                partial(
                    subprocess.run,
                    command,
                    env={**os.environ, **environment},
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
            )
        finally:
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "rm", "--force", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        (sandbox.workspace.parent / "harness.stdout.jsonl").write_text(result.stdout)
        (sandbox.workspace.parent / "harness.stderr.log").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(
                f"opencode exited with {result.returncode}; inspect harness.stderr.log"
            )
        outputs = []
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # upstream installer emits status before CLI JSONL
            if event.get("type") == "text":
                outputs.append(event["part"]["text"])
        if not outputs:
            raise RuntimeError(
                "opencode returned no assistant output; inspect harness.stdout.jsonl"
            )
        return AgentResult("\n".join(outputs), "completed")


def _harness_command(name, workspace, environment):
    """Harness command."""
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        name,
        "--network=host",
        "--cpus=2",
        "--memory=2g",
        "--pids-limit=256",
        "--mount",
        f"type=bind,src={workspace},dst=/workspace",
    ]
    # Values go in the subprocess environment, never shell command interpolation.
    command += [argument for key in environment for argument in ("--env", key)]
    command += ["trains-tracing-opencode:local", "python3", "/opt/run_harness.py"]
    return command
