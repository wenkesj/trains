# Native Harvey LAB results -> messages -> native Baseten Loops training.

import json
import os
import time
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import httpx
import typer
from baseten import loops
from baseten.client.managementapi import ApiClient, DeployLoopsCheckpointRequest
from datasets import Dataset, load_from_disk
from recipes.models import DEFAULT_MODEL, load_tokenizer
from recipes.rl import group_advantages, ppo_update, validate_sampling
from trains.datasets import (
    chat_dataset_from_harvey_run,
    datum_dataset_from_chat_dataset,
)
from trains.datum.adapters.baseten import BasetenDatumAdapter

app = typer.Typer()


@app.command()
def prepare(
    runs: list[Path],
    output: Path = Path("var/chats"),
    score_file: str | None = None,
):
    """Import graded native artifacts into reusable conversations with task groups."""
    dataset = chat_dataset_from_harvey_run(runs, score_file=score_file)
    if not isinstance(dataset, Dataset):
        raise TypeError("Expected a materialized Harvey chat dataset")
    group_advantages(list(dataset))
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output))
    typer.echo(f"Saved {len(dataset)} complete conversations to {output}")


@app.command()
def train(
    chats: Path,
    behavior: Annotated[Path, typer.Option(help="Deployment receipt of the collection policy")],
    tokenizer_model: str = DEFAULT_MODEL,
    training_url: str | None = None,
    steps: int = 1,
    learning_rate: float = 1e-5,
    output: Path = Path("var/checkpoint.json"),
):
    """Restore the behavior checkpoint and apply group-centered PPO to all graded attempts.

    Examples:
        Inspect the native training targets before connecting to a provider:

        >>> from trains.datasets import datum_dataset_from_datums
        >>> from trains.datum.adapters.rendering import native_datum
        >>> datum = BasetenDatumAdapter().from_tokens([1, 2], [3, 4])
        >>> row = datum_dataset_from_datums([datum])[0]
        >>> restored = native_datum(json.loads(row["datum_json"]), loops)
        >>> restored.loss_fn_inputs["target_tokens"].data
        [2, 3, 4]
        >>> restored.loss_fn_inputs["weights"].data
        [0.0, 1.0, 1.0]
        >>> BasetenDatumAdapter().from_tokens([], [3])
        Traceback (most recent call last):
            ...
        ValueError: Tokenized prompt and completion must be nonempty
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    policy = json.loads(behavior.read_text())
    dataset = _training_data(chats, policy, tokenizer_model)
    training_url = training_url or policy.get("training_url")
    service = loops.ServiceClient(base_url=training_url, api_key=os.environ["BASETEN_API_KEY"])
    client = None
    try:
        client = service.create_training_client_from_state_with_optimizer(policy["state"])
        metrics = ppo_update(
            list(dataset),
            client,
            loops,
            behavior_model=policy["model"],
            steps=steps,
            learning_rate=learning_rate,
        )
        state = client.save_state("harvey-state").result()
        sampler_name = f"harvey-{uuid4().hex}"
        sampler = client.save_weights_for_sampler(sampler_name).result()
        run_id, checkpoints = _sampler_checkpoint(service, client, sampler_name)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "state": state.path,
                    "sampler": sampler.path,
                    "base_model": policy["base_model"],
                    "rl": metrics,
                    "training_url": training_url,
                    "run_id": run_id,
                    "checkpoint_id": checkpoints[0].id,
                    "sampler_name": checkpoints[0].checkpoint_id,
                },
                indent=2,
            )
            + "\n"
        )
        typer.echo(output.read_text())
    finally:
        if client:
            client.close()


@app.command()
def initialize(
    base_model: str = DEFAULT_MODEL,
    training_url: str | None = None,
    rank: int = 8,
    output: Path = Path("var/initial-checkpoint.json"),
    sampler_name: str | None = None,
):
    """Save initial optimizer state and publish the collection policy."""
    if rank < 1:
        raise ValueError("rank must be positive")
    service = loops.ServiceClient(base_url=training_url, api_key=os.environ["BASETEN_API_KEY"])
    client = service.create_lora_training_client(base_model=base_model, rank=rank)
    try:
        state = client.save_state("harvey-initial-state").result()
        sampler_name = sampler_name or f"harvey-initial-{uuid4().hex}"
        sampler = client.save_weights_for_sampler(sampler_name).result()
        run_id = client.run_id
        if run_id is None:
            raise ValueError("Training provider did not return a run ID")
        checkpoints = [
            item
            for item in service.list_checkpoints(run_id)
            if item.checkpoint_id == sampler_name and item.target == "sampler"
        ]
        if len(checkpoints) != 1:
            raise ValueError("Expected one initial sampler checkpoint")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "state": state.path,
                    "sampler": sampler.path,
                    "base_model": base_model,
                    "training_url": training_url,
                    "checkpoint_id": checkpoints[0].id,
                    "sampler_name": checkpoints[0].checkpoint_id,
                },
                indent=2,
            )
            + "\n"
        )
        typer.echo(output.read_text())
    finally:
        client.close()


@app.command()
def deploy(
    checkpoint: Path,
    model_name: str = typer.Option(...),
    instance_type: str = typer.Option(...),
    hf_secret: str = typer.Option(...),
    control_url: str = "https://api.baseten.co",
    local: bool = False,
    inference_model: str | None = None,
    timeout: float = 900,
    output: Path = Path("var/deployment.json"),
):
    """Publish the exact saved sampler and record its callable model identity."""
    receipt = json.loads(checkpoint.read_text())
    key = os.environ["BASETEN_API_KEY"]
    with httpx.Client(
        base_url=control_url.rstrip("/"),
        headers={"Authorization": f"Api-Key {key}"},
        timeout=60,
    ) as http:
        api = ApiClient(http)
        deployed = api.post_loops_checkpoints_deploy(
            request=DeployLoopsCheckpointRequest(
                checkpoint_ids=[receipt["checkpoint_id"]],
                model_name=model_name,
                instance_type_id=instance_type,
                hf_secret_name=hf_secret,
            )
        )
        deadline = time.monotonic() + timeout
        while True:
            status = api.get_models_deployments_deployment_id(
                model_id=deployed.model_id, deployment_id=deployed.deployment_id
            )
            if status.status == "ACTIVE":
                break
            if "FAIL" in status.status or status.status in {"STOPPED", "DEACTIVATING"}:
                raise RuntimeError(f"Deployment failed: {status.status}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Deployment {deployed.deployment_id} is still {status.status}")
            time.sleep(5)
    prefix = (
        control_url.rstrip("/") if local else f"https://model-{deployed.model_id}.api.baseten.co"
    )
    result = {
        "state": receipt["state"],
        "base_model": receipt["base_model"],
        "training_url": receipt.get("training_url"),
        "model_id": deployed.model_id,
        "deployment_id": deployed.deployment_id,
        "model": inference_model or receipt["sampler_name"],
        "inference_url": f"{prefix}/deployment/{deployed.deployment_id}/sync/v1",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    typer.echo(output.read_text())


def _training_data(chats, policy, tokenizer_model):
    """Training data."""
    conversations = load_from_disk(str(chats))
    if not isinstance(conversations, Dataset):
        raise TypeError("Select one saved chat dataset, not a dataset dictionary")
    group_advantages(list(conversations), behavior_model=policy["model"])
    validate_sampling(list(conversations))

    tokenizer = load_tokenizer(tokenizer_model)
    if tokenizer_model != DEFAULT_MODEL and "<think" in tokenizer.get_chat_template():
        raise ValueError(
            "LAB transcripts omit hidden reasoning; use a non-reasoning policy for PPO"
        )
    dataset = datum_dataset_from_chat_dataset(
        conversations,
        tokenizer,
        BasetenDatumAdapter(),
    )
    return dataset


def _sampler_checkpoint(service, client, sampler_name):
    """Sampler checkpoint."""
    run_id = client.run_id
    if run_id is None:
        raise ValueError("Training provider did not return a run ID")
    checkpoints = [
        item
        for item in service.list_checkpoints(run_id)
        if item.checkpoint_id == sampler_name and item.target == "sampler"
    ]
    if len(checkpoints) != 1:
        raise ValueError("Expected one published sampler checkpoint in the native listing")
    return run_id, checkpoints


if __name__ == "__main__":
    app()
