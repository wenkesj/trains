# Run real coding CLIs and their native integrations in disposable profiles.

import asyncio
import json
import os
import subprocess
from functools import partial
from uuid import uuid4

from codex_tracing.model_proxy import ModelProxy
from codex_tracing.result import AgentResult
from codex_tracing.sandbox import Sandbox
from codex_tracing.training import Policy


async def solve(sandbox: Sandbox, policy: Policy) -> AgentResult:
    """Run the native coding harness against the candidate workspace and policy."""
    name = "trains-tracing-" + uuid4().hex
    with ModelProxy(policy) as proxy:
        environment = {
            "TRAINS_MODEL_URL": proxy.url,
            "TINKER_API_KEY": os.environ["TINKER_API_KEY"],
        }
        environment.update(
            {
                "ARIZE_NONINTERACTIVE": "true",
                "ARIZE_TRACE_ENABLED": "true",
                "ARIZE_PROJECT_NAME": os.environ.get("TRAINS_TRACE_PROJECT", "trains-codex"),
            }
        )
        keys = (
            "ARIZE_BACKEND",
            "ARIZE_API_KEY",
            "ARIZE_SPACE_ID",
            "ARIZE_OTLP_ENDPOINT",
            "PHOENIX_ENDPOINT",
            "PHOENIX_API_KEY",
        )
        environment.update({key: os.environ[key] for key in keys if key in os.environ})
        environment.setdefault("ARIZE_BACKEND", "arize")
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
        return _agent_result(result.returncode, result.stdout)


def _agent_result(returncode: int, stdout: str) -> AgentResult:
    """Preserve native context exhaustion as a graded terminal attempt."""
    outputs, failures = [], []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # Upstream installer emits status before CLI JSONL.
        if not isinstance(event, dict):
            continue
        if event.get("type") == "turn.failed":
            failures.append((event.get("error") or {}).get("message", ""))
        if event.get("type") == "item.completed":
            item = event["item"]
            if item.get("type") == "agent_message":
                outputs.append(item["text"])
    response = "\n".join(outputs) or None
    if failures:
        message = failures[-1]
        if "Requested token count exceeds" in message and "maximum context length" in message:
            return AgentResult(response, "context_limit")
    if returncode or failures:
        raise RuntimeError(f"codex exited with {returncode}; inspect harness.stderr.log")
    if response is None:
        raise RuntimeError("codex returned no assistant output; inspect harness.stdout.jsonl")
    return AgentResult(response, "completed")


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
    command += ["trains-tracing-codex:local", "python3", "/opt/run_harness.py"]
    return command
