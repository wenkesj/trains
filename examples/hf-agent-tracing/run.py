# Fine-tune on licensed Hub OpenHands trajectories using Baseten Loops.

import argparse
import json
import math
from pathlib import Path
from uuid import uuid4

from baseten import loops
from datasets import Dataset, load_from_disk
from recipes.exercism import DEFAULT_TRAIN, EXERCISM_ROOT, Exercise
from recipes.models import DEFAULT_MODEL, LFM3_REVISION, load_tokenizer
from trains.datasets import chat_dataset_from_records, datum_dataset_from_chat_dataset
from trains.datum.adapters.baseten import BasetenDatumAdapter
from trains.datum.adapters.cookbook import CookbookRenderer
from trains.datum.adapters.rendering import native_datum

from collection import collect
from rendering import LIQUID_OPTIONS, LiquidRenderer, load_renderer
from replay import reinforce
from source import ATTRIBUTION, DATASET, LICENSE, REVISION, URL, load_conversations

DESCRIPTION = "Fine-tune on licensed Hub OpenHands trajectories using Baseten Loops."


def prepare(destination: Path, model: str, limit: int = 4):
    """Stream pinned OpenHands conversations and render native Baseten datum rows."""
    conversations = load_conversations(limit)
    tokenizer = load_tokenizer(model)
    renderer = load_renderer(model, tokenizer)
    chats = chat_dataset_from_records(
        {
            **record,
            **(
                {"chat_template_kwargs": LIQUID_OPTIONS}
                if isinstance(renderer, LiquidRenderer)
                else {}
            ),
        }
        for record in conversations
    )
    processor = tokenizer if isinstance(renderer, LiquidRenderer) else CookbookRenderer(renderer)
    datums = datum_dataset_from_chat_dataset(chats, processor, BasetenDatumAdapter())
    _save_source(destination, model, chats, datums)
    return datums


def sft(
    destination: Path,
    model: str,
    steps: int,
    batch_size: int = 1,
    *,
    learning_rate: float = 1e-5,
):
    """Apply supervised updates, then save optimizer state and sampler weights."""
    manifest = json.loads((destination / "source.json").read_text())
    if manifest["model"] != model:
        raise ValueError("Prepared tokenizer differs from the requested training model")
    rows = load_from_disk(str(destination / "datums"))
    if not isinstance(rows, Dataset):
        raise TypeError("Expected a single prepared datum dataset")
    batch = [native_datum(json.loads(row["datum_json"]), loops) for row in rows]
    if not batch:
        raise ValueError("No supervised datums were prepared")
    service = loops.ServiceClient()
    longest = max(sum(len(chunk.tokens) for chunk in datum.model_input.chunks) for datum in batch)
    client = service.create_lora_training_client(
        base_model=model,
        rank=8,
        max_seq_len=((longest + 1023) // 1024) * 1024,
    )
    try:
        metrics = []
        for step in range(steps):
            start = (step * batch_size) % len(batch)
            minibatch = batch[start : start + batch_size]
            forward = client.forward_backward(minibatch, "cross_entropy").result()
            update = client.optim_step(loops.AdamParams(learning_rate=learning_rate)).result()
            metrics.append({"forward_backward": forward.metrics, "optimizer": update.metrics})
        name = "hf-trace-sft-" + uuid4().hex
        state = client.save_state(name).result()
        sampler = client.save_weights_for_sampler(name).result()
        receipt = {
            "base_model": model,
            "state": state.path,
            "model": sampler.path,
            "metrics": metrics,
            "source": manifest,
        }
        (destination / "policy.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    finally:
        client.close()


def main():
    """Prepare data without provisioning GPUs, or explicitly run Baseten SFT."""
    parser = _parser()
    args = _arguments(parser)
    if args.command == "prepare":
        result = prepare(args.artifacts, args.model, args.limit)
        print(f"Prepared {len(result)} Baseten datums")
    elif args.command == "sft":
        print(
            json.dumps(
                sft(
                    args.artifacts,
                    args.model,
                    args.steps,
                    args.batch_size,
                    learning_rate=args.learning_rate,
                ),
                indent=2,
            )
        )
    else:
        policy = json.loads((args.artifacts / "policy.json").read_text())
        service = loops.ServiceClient()
        if args.command == "collect":
            _collect(args, policy, service, parser)
        else:
            if args.collection is None:
                parser.error("rl requires --collection from the collect command")
            updated = reinforce(
                args.collection.expanduser().resolve(),
                policy,
                service,
                loops,
                steps=args.steps,
                learning_rate=args.learning_rate,
            )
            receipt = json.dumps(updated, indent=2) + "\n"
            pending = args.artifacts / "policy.json.tmp"
            pending.write_text(receipt)
            pending.replace(args.artifacts / "policy.json")
            print(receipt, end="")


def _save_source(destination, model, chats, datums):
    """Save source."""
    destination.mkdir(parents=True, exist_ok=True)
    chats.save_to_disk(str(destination / "chats"))
    datums.save_to_disk(str(destination / "datums"))
    (destination / "source.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "revision": REVISION,
                "license": LICENSE,
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "attribution": ATTRIBUTION,
                "url": URL,
                "trajectories": [json.loads(row["metadata_json"]) for row in chats],
                "model": model,
                "model_revision": LFM3_REVISION if model == DEFAULT_MODEL else None,
                "conversations": len(chats),
                "datums": len(datums),
            },
            indent=2,
        )
        + "\n"
    )


def _parser():
    """Parser."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("command", choices=["prepare", "sft", "collect", "rl"])
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--limit", type=int, default=4, help="Number of pinned Hub trajectories")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument(
        "--exercises",
        default=",".join(DEFAULT_TRAIN),
        help="Upstream Exercism slugs, selected before collection",
    )
    parser.add_argument("--repository", type=Path, default=EXERCISM_ROOT)
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--sessions",
        type=Path,
        default=Path("~/.pi/agent/sessions/trains-hf-agent-tracing"),
    )
    parser.add_argument(
        "--collection",
        type=Path,
        help="Collection directory returned by collect; required by rl",
    )
    parser.add_argument("--image", default="trains-pi-harness:0.73.1")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser


def _arguments(parser):
    """Arguments."""
    args = parser.parse_args()
    args.artifacts = args.artifacts.expanduser().resolve()
    if min(args.max_calls, args.max_tokens) < 1:
        parser.error("--max-calls and --max-tokens must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    return args


def _collect(args, policy, service, parser):
    """Collect."""
    if args.attempts < 2:
        parser.error("--attempts must be at least two")
    slugs = args.exercises.split(",")
    if len(slugs) != len(set(slugs)):
        parser.error("--exercises must contain unique upstream slugs")
    exercises = [Exercise.load(args.repository, slug) for slug in slugs]
    renderer = load_renderer(policy["base_model"])
    sampler = service.create_sampling_client(model_path=policy["model"])
    try:
        batch = collect(
            args.artifacts,
            args.sessions.expanduser().resolve(),
            policy,
            sampler,
            renderer,
            image=args.image,
            attempts=args.attempts,
            exercises=exercises,
            max_calls=args.max_calls,
            max_tokens=args.max_tokens,
        )
        print(batch)
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
