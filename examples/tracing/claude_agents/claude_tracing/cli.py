# Isolated claude_agents eval → native Tinker PPO → eval example.

import json
import os
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from recipes.models import DEFAULT_MODEL

from claude_tracing.benchmark import (
    DEFAULT_EVAL,
    DEFAULT_TRAIN,
    EXERCISM_ROOT,
    Exercise,
    validate_split,
)
from claude_tracing.instrumentation import configure
from claude_tracing.training import Policy, initial_policy, train_once

app = typer.Typer()


@app.command("run")
def run(
    output: Annotated[Path, typer.Option(help="Fresh run directory; never overwrites a run.")],
    policy: Annotated[Path, typer.Option(help="Publication receipt from init-policy.")],
    repository: Annotated[
        Path, typer.Option(help="Exercism Git submodule checkout.")
    ] = EXERCISM_ROOT,
    repetitions: Annotated[int, typer.Option(min=2)] = 4,
    train: str = ",".join(DEFAULT_TRAIN),
    evaluation: str = ",".join(DEFAULT_EVAL),
    learning_rate: Annotated[float, typer.Option(min=1e-8)] = 1e-5,
    steps: Annotated[int, typer.Option(min=1)] = 1,
    evaluation_repetitions: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """Evaluate, collect graded attempts, update the policy, and evaluate again."""
    training_ids, evaluation_ids = tuple(train.split(",")), tuple(evaluation.split(","))
    validate_split(training_ids, evaluation_ids)
    if output.exists():
        raise ValueError("Use a fresh output directory")
    original = Policy.model_validate_json(policy.read_text())
    training = [Exercise.load(repository, slug) for slug in training_ids]
    heldout = [Exercise.load(repository, slug) for slug in evaluation_ids]
    backend, solve, provider = _backend_for("trains-claude_agents-" + uuid4().hex)
    output.mkdir(parents=True)
    original.save(output / "initial-policy.json")
    before = backend.collect(
        heldout,
        original,
        output / "before",
        solve,
        repetitions=evaluation_repetitions,
        phase="before",
    )
    collected = backend.collect(
        training, original, output / "train", solve, repetitions=repetitions, phase="train"
    )
    provider.force_flush()
    chats = backend.chats(collected, original)
    if not len(chats) or len(set(chats["trace_id"])) != len(collected["attempts"]):
        raise ValueError("Provider must export every collected attempt before PPO")
    chats.save_to_disk(str(output / "chats"))
    updated, metrics = train_once(chats, original, learning_rate=learning_rate, steps=steps)
    updated.save(output / "updated-policy.json")
    (output / "training.json").write_text(json.dumps(metrics, indent=2) + "\n")
    after = backend.collect(
        heldout, updated, output / "after", solve, repetitions=evaluation_repetitions, phase="after"
    )
    provider.force_flush()
    report = {
        "harness": "claude_agents",
        "training": metrics,
        "before": [r["reward"] for r in before["attempts"]],
        "after": [r["reward"] for r in after["attempts"]],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


@app.command("init-policy")
def initialize(
    output: Path,
    base_model: str = DEFAULT_MODEL,
    training_url: str = "http://127.0.0.1:8382",
    inference_url: str = "http://127.0.0.1:8382/v1",
    renderer: str | None = None,
) -> None:
    """Save initial optimizer state and publish the collection policy."""
    if output.exists():
        raise ValueError("Policy receipt already exists")
    policy = initial_policy(
        base_model=base_model,
        tokenizer_model=base_model,
        training_url=training_url,
        inference_url=inference_url,
        renderer=renderer,
        rendering_options={
            "enable_thinking": False,
            **({"preserve_thinking": True} if base_model == DEFAULT_MODEL else {}),
        },
    )
    policy.save(output)


def _backend_for(project: str):
    """Configure the native trace provider and return its collector and harness."""
    selected = os.environ.get("TRAINS_TRACE_PROVIDER", "langfuse")
    if selected == "langsmith":
        return _langsmith_backend(project)
    if selected != "langfuse":
        raise ValueError("Select TRAINS_TRACE_PROVIDER=langfuse or langsmith")
    project = os.environ.get("TRAINS_TRACE_PROJECT", project)
    os.environ["TRAINS_TRACE_PROJECT"] = project
    from langfuse import Langfuse

    from .agent import solve
    from .langfuse import LangfuseBackend

    provider = configure(project=project, endpoint=None)
    client = Langfuse(tracer_provider=provider)
    if not client.auth_check():
        raise ValueError("Langfuse credentials failed authentication")
    return LangfuseBackend(client), solve, provider


def _langsmith_backend(project: str):
    """Configure the native trace provider and return its collector and harness."""
    project = os.environ.get("TRAINS_TRACE_PROJECT", project)
    project = os.environ.get("LANGSMITH_PROJECT", project)
    os.environ["TRAINS_TRACE_PROJECT"] = project
    from langsmith import Client

    from .agent import solve
    from .langsmith import LangSmithBackend

    url = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/")
    agent_id, agent_environment = (
        os.getenv("LANGSMITH_AGENT_ID"),
        os.getenv("LANGSMITH_AGENT_ENVIRONMENT"),
    )
    headers = {"x-api-key": os.environ["LANGSMITH_API_KEY"]}
    if agent_id or agent_environment:
        if not agent_id or not agent_environment or not os.getenv("LANGSMITH_PROJECT"):
            raise ValueError(
                "Agent workspaces require agent ID, environment, and compatibility project name"
            )
        headers.update(
            {"Langsmith-Agent-Id": agent_id, "Langsmith-Agent-Environment": agent_environment}
        )
    else:
        headers["Langsmith-Project"] = project
    os.environ["LANGSMITH_PROJECT"] = project
    provider = configure(project=project, endpoint=url + "/otel/v1/traces", headers=headers)
    return LangSmithBackend(Client(auto_batch_tracing=False), project=project), solve, provider
