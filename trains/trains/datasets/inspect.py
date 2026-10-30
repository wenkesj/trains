import json
import math
from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

from datasets import Dataset, IterableDataset
from trains.datasets.chat import FEATURES, chat_row
from trains.datasets.generator import from_generator
from trains.formats.chat import chat_completion_record
from trains.formats.sampling import sampling_parameters
from trains.types import ChatRow

try:
    from inspect_ai import log as _inspect_log
except ImportError as error:
    _inspect_log = None
    _INSPECT_IMPORT_ERROR = error
else:
    _INSPECT_IMPORT_ERROR = None


def chat_dataset_from_inspect_log(
    path: str | Path | Iterable[str | Path],
    *,
    model: str | None = None,
    scorer: str | None = None,
    min_score: float | None = None,
    streaming: bool = False,
) -> Dataset | IterableDataset:
    """Read completed native Inspect logs, retaining agent calls and optional scores.

    Requires trains-sdk[inspect]. Collect with --log-model-api. Each exact Chat
    Completions request/response supervises its final assistant only. Media
    attachments and pooled payloads are resolved by Inspect's own reader.

    Args:
        path: Native artifact file, directory, or collection of source paths.
        model: Optional evaluated-model identity; judge calls are excluded.
        scorer: Scorer name to select when a sample has multiple native scores.
        min_score: Explicit lower score bound; omit it to retain failed attempts for RL.
        streaming: Return a lazy IterableDataset instead of materializing all rows.

    Returns:
        Complete evaluated-model chat rows with native scores and task identities.
    """
    if _inspect_log is None:
        raise ImportError(
            "Inspect log support requires inspect-ai. Install it with pip install 'trains-sdk[inspect]'."
        ) from _INSPECT_IMPORT_ERROR
    paths = [path] if isinstance(path, (str, Path)) else path
    paths = tuple(sorted({Path(item).resolve() for item in paths}))
    if not paths:
        raise ValueError("No Inspect logs selected")
    return from_generator(
        partial(_inspect_chats, paths, model, scorer, min_score), FEATURES, streaming=streaming
    )


def _score_value(sample: Any, scorer: str | None) -> float:
    """Normalize a finite numeric or correct/incorrect Inspect score."""
    scores = sample.scores or {}
    if scorer is None:
        if len(scores) != 1:
            raise ValueError("Select --scorer when a sample has zero or multiple scores")
        scorer = next(iter(scores))
    value = scores[scorer].value
    if value in ("C", "I"):
        return float(value == "C")
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Select a scalar numeric or C/I score")
    return float(value)


def _records_from_log(
    path: Path,
    *,
    model: str | None = None,
    scorer: str | None = None,
    min_score: float | None = None,
) -> Iterator[dict[str, Any]]:
    # The official reader resolves pooled model-call references and media attachments.
    """Read evaluated-model requests and resolve native Inspect attachments."""
    if _inspect_log is None:
        raise ImportError(
            "Inspect log support requires inspect-ai. Install it with pip install 'trains-sdk[inspect]'."
        ) from _INSPECT_IMPORT_ERROR
    log = _inspect_log.read_eval_log(str(path), resolve_attachments=True)
    if log.status != "success":
        raise ValueError("Select a successfully completed Inspect evaluation")
    model = model or log.eval.model
    count = 0
    for sample in log.samples or []:
        if sample.error or sample.limit:
            continue
        reward = _score_value(sample, scorer) if sample.scores or scorer else None
        if min_score is not None and (reward is None or reward < min_score):
            continue
        calls = [
            event
            for event in sample.events
            if event.event == "model"
            and event.model == model
            and getattr(event, "origin", None) in (None, "agent")
        ]
        if not calls:
            raise ValueError(f"Sample {sample.id}: no model events for {model!r}")
        for index, event in enumerate(calls):
            call = event.call
            if event.error or call is None or call.error or call.response is None:
                raise ValueError(f"Sample {sample.id}: missing successful raw model call")
            if "messages" not in call.request or "choices" not in call.response:
                raise ValueError(
                    "This example imports Chat Completions calls; add an adapter for other APIs"
                )
            request = _restore_images(call.request, event.input)
            record = chat_completion_record(request, call.response)
            count += 1
            yield _inspect_record(record, request, log.eval.eval_id, sample, index, reward)

    if not count:
        raise ValueError(
            "No accepted evaluated-model calls; lower --min-score or collect more samples"
        )


def _restore_images(request: Mapping[str, Any], messages: Iterable[Any]) -> dict[str, Any]:
    """Restore Inspect-redacted image bytes from its resolved model input."""
    restored = deepcopy(dict(request))
    images = [
        part["image_url"]
        for message in restored.get("messages", [])
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]
    if not any(image.get("url") == "<base64-data-removed>" for image in images):
        return restored
    native = [
        part.image
        for message in messages
        if isinstance(message.content, list)
        for part in message.content
        if part.type == "image"
    ]
    if len(native) != len(images):
        raise ValueError("Cannot align redacted request images with resolved Inspect model input")
    for image, source in zip(images, native, strict=True):
        if image.get("url") == "<base64-data-removed>":
            if not source or source == "<base64-data-removed>":
                raise ValueError("Inspect log is missing resolved image data")
            image["url"] = source
    return restored


def _inspect_record(
    record: Mapping[str, Any],
    request: Mapping[str, Any],
    eval_id: str,
    sample: Any,
    index: int,
    reward: float | None,
) -> dict[str, Any]:
    """Convert a complete Inspect model call into a portable chat record."""
    return {
        **record,
        "reward": reward,
        "model_id": request.get("model"),
        "metadata_json": json.dumps({"sampling": sampling_parameters(request)}),
        "trace_id": f"{eval_id}:{sample.id}:{sample.epoch}",
        "group_id": str(sample.id),
        "span_id": str(index),
    }


def _inspect_chats(
    paths: tuple[Path, ...],
    model: str | None,
    scorer: str | None,
    min_score: float | None,
) -> Iterator[ChatRow]:
    """Yield evaluated-model chats from native Inspect logs with task grades."""
    for path in paths:
        for record in _records_from_log(path, model=model, scorer=scorer, min_score=min_score):
            yield chat_row(record)
