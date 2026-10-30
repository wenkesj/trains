# Collect native Pi sessions from the SFT sampler and grade coding attempts.

import json
import os
import subprocess
from uuid import uuid4

from recipes.verification import UNITTEST_VERIFIER

from bridge import SamplerBridge

PROMPT = (
    "Fix solution.py: implement count_vowels(text), returning the number of ASCII "
    "vowels (a,e,i,o,u) in the input, case insensitive. Read the existing file, "
    "write the fix, and briefly report completion."
)
STUB = "def count_vowels(text):\n    return 0\n"


IMAGE = "trains-pi-harness:0.73.1"


def collect_one(
    directory,
    session_root,
    policy,
    sampler,
    renderer,
    *,
    image=IMAGE,
    max_calls=20,
    max_tokens=2048,
    exercise=None,
):
    """Run the actual Pi CLI with a private profile and a pinned token sampler."""
    image = subprocess.check_output(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
    ).strip()
    directory.mkdir(parents=True, exist_ok=False)
    workspace = directory / "workspace"
    prompt = _workspace(workspace, exercise)
    profile = directory / "profile"
    profile.mkdir()
    session_dir = session_root / directory.name
    session_dir.mkdir(parents=True, exist_ok=False)
    with SamplerBridge(
        sampler, renderer, policy["model"], max_calls=max_calls, max_tokens=max_tokens
    ) as bridge:
        _profile(profile, policy, bridge.url, max_tokens)
        container = "trains-pi-" + uuid4().hex
        command = _harness_command(image, container, workspace, profile, session_dir, prompt)
        status, stdout, stderr = _run_harness(command, workspace, container)
        (directory / "stdout.jsonl").write_text(stdout)
        (directory / "stderr.log").write_text(stderr)
    reward, grading = _grade(
        workspace, image=image, exercise=exercise, verifier=directory / "verifier"
    )
    if status:
        reward = 0.0
    return _save_receipt(
        directory, session_dir, policy, bridge, image, exercise, status, reward, grading, max_tokens
    )


def collect(
    destination,
    session_root,
    policy,
    sampler,
    renderer,
    *,
    image=IMAGE,
    attempts=4,
    exercises=None,
    max_calls=20,
    max_tokens=2048,
):
    """Keep every attempt, including failures, in one immutable collection batch."""
    if attempts < 2:
        raise ValueError("RL requires at least two independent attempts per task")
    batch = destination / ("collection-" + uuid4().hex)
    batch.mkdir(parents=True)
    receipts = []
    for exercise in exercises if exercises is not None else [None]:
        for _ in range(attempts):
            receipts.append(
                collect_one(
                    batch / uuid4().hex,
                    session_root,
                    policy,
                    sampler,
                    renderer,
                    image=image,
                    exercise=exercise,
                    max_calls=max_calls,
                    max_tokens=max_tokens,
                )
            )
    manifest = {"model": policy["model"], "attempts": [r["id"] for r in receipts]}
    (batch / "collection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return batch


def _workspace(workspace, exercise):
    """Copy original starter files and describe their exact candidate paths."""
    if exercise is None:
        workspace.mkdir()
        (workspace / "solution.py").write_text(STUB)
        return PROMPT
    exercise.candidate(workspace)
    solutions = ", ".join(f"/workspace/{name}" for name in exercise.solutions)
    return (
        f"Read /workspace/INSTRUCTIONS.md and the starter solution files: {solutions}. "
        "Implement the Python exercise by editing those files, then briefly report "
        "completion. Available tools are read, edit, and write. "
        "Hidden upstream tests are run separately."
    )


def _profile(profile, policy, url, max_tokens):
    """Write an isolated Pi provider profile with only the local bridge credential."""
    model = {
        "id": "policy",
        "name": "SFT policy",
        "reasoning": True,
        "input": ["text"],
        "contextWindow": 128000
        if policy.get("base_model", "").startswith("LiquidAI/LFM3-")
        else 32768,
        "maxTokens": max_tokens,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    }
    provider = {
        "baseUrl": url,
        "api": "openai-completions",
        "apiKey": "local-bridge",
        "models": [model],
    }
    (profile / "models.json").write_text(json.dumps({"providers": {"trains": provider}}))


def _harness_command(image, container, workspace, profile, session_dir, prompt):
    """Keep native CLI options and Docker resource boundaries explicit."""
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container,
        "--network=host",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=2g",
        "--pids-limit=256",
        "--cpus=2",
        "--tmpfs=/tmp:rw,nosuid,nodev,size=256m",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
    ]
    for source, target in [
        (workspace, "/workspace"),
        (profile, "/profile"),
        (session_dir, "/sessions"),
    ]:
        command += ["--mount", f"type=bind,src={source.resolve()},dst={target}"]
    return command + [
        "--workdir=/workspace",
        "--env=HOME=/profile",
        "--env=PI_CODING_AGENT_DIR=/profile",
        "--env=PI_TELEMETRY=0",
        image,
        "--provider",
        "trains",
        "--model",
        "policy",
        "--thinking",
        "medium",
        "--session-dir",
        "/sessions",
        "--tools",
        "read,edit,write",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-context-files",
        "--offline",
        "--mode",
        "json",
        "--print",
        prompt,
    ]


def _run_harness(command, workspace, container):
    """Retain native process output and always remove the owned container."""
    try:
        result = subprocess.run(
            command, cwd=workspace, check=False, capture_output=True, text=True, timeout=600
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        stdout = (
            (error.stdout or b"").decode()
            if isinstance(error.stdout, bytes)
            else error.stdout or ""
        )
        return -1, stdout, "Pi timed out"
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container], check=False, capture_output=True, timeout=30
        )


def _save_receipt(
    directory, session_dir, policy, bridge, image, exercise, status, reward, grading, max_tokens
):
    """Persist every native action and grade, including failed attempts."""
    receipt = {
        "id": directory.name,
        "group_id": exercise.slug if exercise is not None else "count-vowels",
        "benchmark": exercise.manifest() if exercise is not None else None,
        "model": policy["model"],
        "reward": reward,
        "exit_code": status,
        "grading": grading,
        "session_dir": str(session_dir),
        "calls": bridge.calls,
        "errors": bridge.errors,
        "harness_image": image,
        "sampling": {"temperature": 1, "top_p": 1, "top_k": -1, "max_tokens": max_tokens},
    }
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _grade(workspace, *, image=IMAGE, exercise=None, verifier=None):
    """Run hidden assertions in a separate container without network access."""
    program = (
        "import runpy; f=runpy.run_path('solution.py')['count_vowels']; "
        "assert all(f(s)==n for s,n in [('',0),('xyz',0),('AEIOU',5),"
        "('Hello World!',3),('queue',4),('naïve',2)])"
    )
    if exercise is not None:
        if verifier is None:
            raise ValueError("Native exercise grading requires a private verifier directory")
        exercise.verifier(workspace, verifier)
        workspace, program = verifier, UNITTEST_VERIFIER
    container = "trains-pi-grader-" + uuid4().hex
    try:
        result = subprocess.run(
            _grader_command(workspace, image, container, program),
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=60 if exercise is not None else 10,
        )
        if result.returncode >= 125:
            raise RuntimeError("Native verifier container failed to start")
        if exercise is not None:
            if result.returncode:
                raise RuntimeError("Native unittest verifier did not complete")
            grade = json.loads(result.stdout.strip().splitlines()[-1])
            if not grade["tests"]:
                raise ValueError("Native verifier ran no tests")
            return float(grade["reward"]), json.dumps({**grade, "stderr": result.stderr})
        return float(result.returncode == 0), result.stderr
    except subprocess.TimeoutExpired:
        return 0.0, "grader timed out"
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container], check=False, capture_output=True, timeout=30
        )


def _grader_command(workspace, image, container, program):
    """Mount only verifier inputs into a resource-bounded, network-free process."""
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        container,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=256m",
        "--pids-limit=64",
        "--cpus=1",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--mount",
        f"type=bind,src={workspace.resolve()},dst=/workspace,readonly",
        "--workdir=/workspace",
        "--entrypoint=python3",
        image,
        "-I",
        "-c",
        program,
    ]
