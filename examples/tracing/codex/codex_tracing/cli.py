# Isolated codex eval → native Tinker PPO → eval example.

import json
import os
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from recipes.models import DEFAULT_MODEL

from codex_tracing.benchmark import (
    DEFAULT_EVAL,
    DEFAULT_TRAIN,
    EXERCISM_ROOT,
    Exercise,
    validate_split,
)
from codex_tracing.instrumentation import configure
from codex_tracing.training import Policy, initial_policy, train_once

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
    backend, solve, provider = _backend_for("trains-codex-" + uuid4().hex)
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
        "harness": "codex",
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
    project = os.environ.get("TRAINS_TRACE_PROJECT", project)
    os.environ["TRAINS_TRACE_PROJECT"] = project
    from arize import ArizeClient

    from .agent import solve
    from .arize_ax import AXBackend

    headers = {"api_key": os.environ["ARIZE_API_KEY"], "space_id": os.environ["ARIZE_SPACE_ID"]}
    provider = configure(
        project=project,
        endpoint=os.getenv("ARIZE_OTLP_ENDPOINT", "https://otlp.arize.com/v1/traces"),
        headers=headers,
    )
    return (
        AXBackend(ArizeClient(), space=os.environ["ARIZE_SPACE_ID"], project=project),
        solve,
        provider,
    )
