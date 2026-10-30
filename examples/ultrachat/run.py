# Train directly from an existing HF messages dataset with native Fireworks.

import json
import math
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import typer
from datasets import load_dataset, load_from_disk
from filelock import FileLock
from fireworks import Fireworks
from fireworks.training import sdk
from recipes.models import DEFAULT_MODEL, load_tokenizer
from trains.datasets import datum_dataset_from_chat_dataset
from trains.datum.adapters.fireworks import FireworksDatumAdapter
from trains.datum.adapters.rendering import native_datum

app = typer.Typer()
REVISION = "b7fe606ecdbf71e8537946a8d9de5ccf0f6da48b"
DATA_DIR = Path("var/data/ultrachat")


@app.command()
def cache(data_dir: Path = DATA_DIR, tokenizer_model: str = DEFAULT_MODEL):
    """Cache both dataset splits and the matching tokenizer for offline training."""
    _tokenizer(tokenizer_model)
    for split in ("train_sft", "test_sft"):
        dataset = _cached_split(split, data_dir)
        typer.echo(f"{split}: {len(dataset)} conversations cached at {data_dir / REVISION / split}")


@app.command()
def train(
    base_model: str = DEFAULT_MODEL,
    tokenizer_model: str = DEFAULT_MODEL,
    training_url: str | None = None,
    samples: int = 32,
    steps: int = 1,
    rank: int = 8,
    learning_rate: float = 1e-5,
    checkpoint: Path | None = None,
    offset: int = 0,
    evaluation_samples: int = 0,
    data_dir: Path = DATA_DIR,
    offline: bool = False,
    output: Path = Path("var/checkpoint.json"),
):
    """Train assistant tokens from complete UltraChat conversations with Fireworks.

    Examples:
        Inspect the native training targets before connecting to a provider:

        >>> from trains.datasets import datum_dataset_from_datums
        >>> from trains.datum.adapters.rendering import native_datum
        >>> datum = FireworksDatumAdapter().from_tokens([1, 2], [3, 4])
        >>> row = datum_dataset_from_datums([datum])[0]
        >>> restored = native_datum(json.loads(row["datum_json"]), sdk)
        >>> restored.loss_fn_inputs["target_tokens"].data
        [2, 3, 4]
        >>> restored.loss_fn_inputs["weights"].data
        [0.0, 1.0, 1.0]
        >>> FireworksDatumAdapter().from_tokens([], [3])
        Traceback (most recent call last):
            ...
        ValueError: Tokenized prompt and completion must be nonempty
    """
    _validate_training(samples, steps, rank, offset, evaluation_samples, learning_rate)
    previous = json.loads(checkpoint.read_text()) if checkpoint else None
    if previous and previous["base_model"] != base_model:
        raise ValueError("Checkpoint base model differs from --base-model")
    tokenizer = _tokenizer(tokenizer_model, offline)
    batch, count = _batch("train_sft", samples, offset, tokenizer, data_dir, offline)
    heldout, heldout_count = (
        _batch("test_sft", evaluation_samples, 0, tokenizer, data_dir, offline)
        if evaluation_samples
        else ([], 0)
    )
    training_url = training_url or (previous or {}).get("training_url")
    service = sdk.ServiceClient(base_url=training_url, api_key=os.environ["FIREWORKS_API_KEY"])
    try:
        client = _training_client(service, base_model, rank, previous)
        history = _train_steps(client, batch, heldout, steps, learning_rate)
        state = client.save_state("ultrachat-state").result()
        sampler = client.save_weights_for_sampler("ultrachat-sampler").result()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "state": state.path,
                    "sampler": sampler.path,
                    "base_model": base_model,
                    "rank": previous.get("rank", rank) if previous else rank,
                    "training_url": training_url,
                    "dataset_revision": REVISION,
                    "training_slice": {"split": "train_sft", "offset": offset, "samples": samples},
                    "training_tokens": count,
                    "evaluation_tokens": heldout_count,
                    "evaluation_slice": {
                        "split": "test_sft",
                        "offset": 0,
                        "samples": evaluation_samples,
                    },
                    "learning_rate": learning_rate,
                    "history": history,
                    "job_id": getattr(service, "managed_trainer_job_id", None),
                },
                indent=2,
            )
            + "\n"
        )
        typer.echo(output.read_text())
    finally:
        service.close()


@app.command()
def deploy(
    checkpoint: Path,
    account_id: str = typer.Option(...),
    model_id: str = typer.Option(...),
    deployment_id: str = typer.Option(...),
    job_id: str | None = None,
    checkpoint_name: str | None = None,
    control_url: str | None = None,
    timeout: float = 900,
    output: Path = Path("var/deployment.json"),
):
    """Publish the exact saved sampler and record its callable model identity."""
    receipt = json.loads(checkpoint.read_text())
    sampler = receipt["sampler"]
    job_id = job_id or receipt.get("job_id")
    if job_id is None and sampler.startswith("local/run-"):
        job_id = sampler.split("/")[1].removeprefix("run-")
    if not job_id:
        raise ValueError("Pass --job-id for the hosted trainer that produced this checkpoint")
    key = os.environ["FIREWORKS_API_KEY"]
    control = sdk.FireworksClient(api_key=key, base_url=control_url)
    checkpoints = [
        item
        for item in control.list_checkpoints(job_id)
        if item.get("promotable")
        and (
            item["name"] == checkpoint_name
            if checkpoint_name
            else item["name"].endswith("/" + sampler.rsplit("/", 1)[-1])
        )
    ]
    if len(checkpoints) != 1:
        raise ValueError("Pass --checkpoint-name to select exactly one promotable checkpoint")
    promoted = control.promote_checkpoint(
        name=checkpoints[0]["name"],
        output_model_id=model_id,
        base_model=receipt["base_model"],
    )
    deployment = _create_deployment(key, control_url, account_id, deployment_id, promoted, timeout)
    inference = (
        control_url.rstrip("/") + "/inference"
        if control_url
        else "https://api.fireworks.ai/inference"
    )
    result = {
        "model": promoted["name"],
        "deployment_id": deployment.name,
        "inference_url": inference,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    typer.echo(output.read_text())


def _cached_split(split, data_dir, offline=False):
    """Read a revision-pinned local Arrow dataset, downloading only on a cache miss."""
    destination = data_dir / REVISION / split
    if destination.exists():
        return load_from_disk(str(destination))
    if offline:
        raise ValueError(
            f"Missing cached {split}; run `python run.py cache --data-dir {data_dir}` first"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(destination.parent / f".{split}.lock"):
        if not destination.exists():
            dataset = load_dataset(
                "HuggingFaceH4/ultrachat_200k",
                data_files={split: f"data/{split}-*.parquet"},
                split=split,
                revision=REVISION,
            )
            with TemporaryDirectory(dir=destination.parent, prefix=f".{split}-") as temporary:
                staged = Path(temporary) / "dataset"
                dataset.save_to_disk(str(staged))
                staged.rename(destination)
    return load_from_disk(str(destination))


def _tokenizer(model, offline=False):
    """Prefer cached tokenizer files and download only when permitted and missing."""
    try:
        return load_tokenizer(model, local_files_only=True)
    except OSError:
        if offline:
            raise
        return load_tokenizer(model)


def _batch(split, samples, offset, tokenizer, data_dir=DATA_DIR, offline=False):
    """Render full conversations and normalize over assistant tokens once."""
    chats = _cached_split(split, data_dir, offline)
    if offset + samples > len(chats):
        raise ValueError("Requested conversation slice exceeds the dataset split")
    chats = chats.select(range(offset, offset + samples))
    data = datum_dataset_from_chat_dataset(chats, tokenizer, FireworksDatumAdapter())
    batch = [native_datum(json.loads(row["datum_json"]), sdk) for row in data]
    count = sum(sum(d.loss_fn_inputs["weights"].data) for d in batch)
    if not count:
        raise ValueError("No supervised tokens")
    for datum in batch:
        weights = datum.loss_fn_inputs["weights"]
        datum.loss_fn_inputs["weights"] = sdk.TensorData(
            data=[w / count for w in weights.data], dtype="float32", shape=weights.shape
        )
    return batch, count


def _nll(client, batch):
    """Measure assistant-token mean negative log likelihood without an update."""
    scored = client.forward(batch, "cross_entropy").result()
    value = math.fsum(
        -weight * logprob
        for datum, output in zip(batch, scored.loss_fn_outputs, strict=True)
        for weight, logprob in zip(
            datum.loss_fn_inputs["weights"].data, output["logprobs"].data, strict=True
        )
        if weight
    )
    if not math.isfinite(value):
        raise ValueError("Evaluation returned nonfinite assistant log probabilities")
    return value


def _train_steps(client, batch, heldout, steps, learning_rate):
    """Train steps."""
    history = [{"step": 0, "heldout_nll": _nll(client, heldout)}] if heldout else []
    for step in range(1, steps + 1):
        client.forward_backward(batch, "cross_entropy").result()
        optimized = client.optim_step(sdk.AdamParams(learning_rate=learning_rate)).result()
        if heldout:
            history.append(
                {
                    "step": step,
                    "heldout_nll": _nll(client, heldout),
                    "optimizer": optimized.metrics,
                }
            )
    return history


def _create_deployment(key, control_url, account_id, deployment_id, promoted, timeout):
    """Create deployment."""
    with Fireworks(api_key=key, base_url=control_url) as cloud:
        deployment = cloud.deployments.create(
            account_id=account_id,
            deployment_id=deployment_id,
            base_model=promoted["name"],
        )
        deadline = time.monotonic() + timeout
        while deployment.state != "READY":
            if (deployment.state and "FAIL" in deployment.state) or deployment.state in {
                "DELETED",
                "DELETING",
            }:
                raise RuntimeError(f"Deployment failed: {deployment.state}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Deployment {deployment_id} is still {deployment.state}")
            time.sleep(5)
            deployment = cloud.deployments.get(deployment_id, account_id=account_id)
    return deployment


def _validate_training(samples, steps, rank, offset, evaluation_samples, learning_rate):
    """Validate training."""
    if min(samples, steps, rank) < 1 or min(offset, evaluation_samples) < 0:
        raise ValueError(
            "Positive samples, steps and rank; nonnegative offsets and evaluation size required"
        )
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")


def _training_client(service, base_model, rank, previous):
    """Training client."""
    client = service.create_lora_training_client(
        base_model=base_model,
        rank=previous.get("rank", rank) if previous else rank,
        train_unembed=False,
    )
    if previous:
        client.load_state_with_optimizer(previous["state"]).result()
    return client


if __name__ == "__main__":
    app()
