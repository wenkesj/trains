# Official Harbor jobs -> messages -> native Tinker-compatible training.

import json
import os
from enum import Enum
from pathlib import Path
from typing import Annotated

import tinker
import typer
from datasets import Dataset, load_from_disk
from recipes.models import DEFAULT_MODEL, load_tokenizer
from recipes.rl import group_advantages, ppo_update, validate_sampling
from tinker import types
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from trains.datasets import (
    chat_dataset_from_harbor_job,
    datum_dataset_from_chat_dataset,
)
from trains.datum.adapters.cookbook import CookbookRenderer
from trains.datum.adapters.spindle import SpindleDatumAdapter
from trains.datum.adapters.tinker import TinkerDatumAdapter


class Provider(str, Enum):
    """The native training protocol used for state restore and sampler publication."""

    tinker = "tinker"
    spindle = "spindle"


app = typer.Typer()


@app.command()
def prepare(
    job: Path,
    output: Path = Path("var/chats"),
    reward_name: str | None = None,
):
    """Import graded native artifacts into reusable conversations with task groups."""
    dataset = chat_dataset_from_harbor_job(job, reward_name=reward_name)
    group_advantages(list(dataset))
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output))
    typer.echo(f"Saved {len(dataset)} complete conversations to {output}")


@app.command()
def train(
    chats: Path,
    behavior: Annotated[Path, typer.Option(help="Deployment receipt of the collection policy")],
    tokenizer_model: str = DEFAULT_MODEL,
    renderer: str | None = None,
    training_url: str | None = None,
    provider: Provider = Provider.tinker,
    steps: int = 1,
    learning_rate: float = 1e-5,
    output: Path = Path("var/checkpoint.json"),
):
    """Restore the behavior checkpoint and apply group-centered PPO to all graded attempts.

    Examples:
        Inspect the native training targets before connecting to a provider:

        >>> from trains.datasets import datum_dataset_from_datums
        >>> from trains.datum.adapters.rendering import native_datum
        >>> datum = TinkerDatumAdapter().from_tokens([1, 2], [3, 4])
        >>> row = datum_dataset_from_datums([datum])[0]
        >>> restored = native_datum(json.loads(row["datum_json"]), types)
        >>> restored.loss_fn_inputs["target_tokens"].data
        [2, 3, 4]
        >>> restored.loss_fn_inputs["weights"].data
        [0.0, 1.0, 1.0]
        >>> TinkerDatumAdapter().from_tokens([], [3])
        Traceback (most recent call last):
            ...
        ValueError: Tokenized prompt and completion must be nonempty
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    policy = json.loads(behavior.read_text())
    dataset = _training_data(chats, policy, tokenizer_model, renderer, provider)
    training_url = training_url or policy.get("training_url") or os.environ.get("TINKER_BASE_URL")
    service = tinker.ServiceClient(base_url=training_url, api_key=os.environ["TINKER_API_KEY"])
    try:
        client = service.create_training_client_from_state_with_optimizer(policy["state"])
        metrics = ppo_update(
            list(dataset),
            client,
            types,
            behavior_model=policy["model"],
            steps=steps,
            learning_rate=learning_rate,
        )
        state = client.save_state("harbor-state").result()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "state": state.path,
                    "base_model": policy["base_model"],
                    "rl": metrics,
                    "provider": provider.value,
                    "training_url": training_url,
                },
                indent=2,
            )
            + "\n"
        )
        typer.echo(output.read_text())
    finally:
        service.close("success").result()


@app.command()
def deploy(
    checkpoint: Annotated[Path | None, typer.Argument()] = None,
    inference_url: str = typer.Option(..., help="Checkpoint-aware OpenAI-compatible chat gateway"),
    training_url: str | None = None,
    base_model: str | None = None,
    provider: Provider = Provider.tinker,
    rank: int = 8,
    output: Path = Path("var/deployment.json"),
):
    """Publish the exact saved sampler and record its callable model identity."""
    if checkpoint is not None and base_model is not None:
        raise ValueError("Select a saved checkpoint or a fresh base model")
    if checkpoint is None and base_model is None:
        base_model = DEFAULT_MODEL
    if checkpoint is None and (not base_model or rank < 1):
        raise ValueError("Pass a checkpoint or --base-model with a positive --rank")
    receipt = json.loads(checkpoint.read_text()) if checkpoint else {"provider": provider.value}
    url = training_url or receipt.get("training_url") or os.environ.get("TINKER_BASE_URL")
    service = tinker.ServiceClient(base_url=url, api_key=os.environ["TINKER_API_KEY"])
    try:
        client = (
            service.create_training_client_from_state(receipt["state"])
            if checkpoint
            else service.create_lora_training_client(
                base_model=base_model, rank=rank, train_unembed=False
            )
        )
        state_path = (
            receipt["state"] if checkpoint else client.save_state("publication-state").result().path
        )
        sampler = client.save_weights_for_sampler("harbor-sampler").result()
        service.create_sampling_client(model_path=sampler.path)
        result = {
            "model": sampler.path,
            "state": state_path,
            "base_model": receipt["base_model"] if checkpoint else base_model,
            "training_url": url,
            "inference_url": inference_url.rstrip("/"),
            "provider": receipt.get("provider", "tinker"),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        typer.echo(output.read_text())
    finally:
        service.close("success").result()


def _training_data(chats, policy, tokenizer_model, renderer, provider):
    """Training data."""
    conversations = load_from_disk(str(chats))
    if not isinstance(conversations, Dataset):
        raise TypeError("Select one saved chat dataset, not a dataset dictionary")
    group_advantages(list(conversations), behavior_model=policy["model"])
    validate_sampling(list(conversations))

    processor = (
        CookbookRenderer(
            get_renderer(renderer, get_tokenizer(tokenizer_model), model_name=tokenizer_model)
        )
        if renderer
        else load_tokenizer(tokenizer_model)
    )
    dataset = datum_dataset_from_chat_dataset(
        conversations,
        processor,
        TinkerDatumAdapter() if provider == Provider.tinker else SpindleDatumAdapter(),
    )
    return dataset


if __name__ == "__main__":
    app()
