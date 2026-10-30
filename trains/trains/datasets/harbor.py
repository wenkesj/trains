import json
import math
from collections.abc import Iterable, Iterator, Mapping
from functools import partial
from pathlib import Path
from typing import Any, Literal, overload

from datasets import Dataset, IterableDataset
from trains.datasets.chat import FEATURES, chat_row
from trains.datasets.generator import from_generator
from trains.formats.atif import ATIFSpec
from trains.formats.records import validate_record
from trains.formats.sampling import sampling_parameters
from trains.types import ChatRow


def read_object(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject non-object artifact payloads."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


@overload
def chat_dataset_from_harbor_job(
    path: str | Path | Iterable[str | Path],
    *,
    spec: ATIFSpec | None = None,
    reward_name: str | None = None,
    min_reward: float | None = None,
    streaming: Literal[False] = False,
) -> Dataset: ...


@overload
def chat_dataset_from_harbor_job(
    path: str | Path | Iterable[str | Path],
    *,
    spec: ATIFSpec | None = None,
    reward_name: str | None = None,
    min_reward: float | None = None,
    streaming: Literal[True],
) -> IterableDataset: ...


@overload
def chat_dataset_from_harbor_job(
    path: str | Path | Iterable[str | Path],
    *,
    spec: ATIFSpec | None = None,
    reward_name: str | None = None,
    min_reward: float | None = None,
    streaming: bool = False,
) -> Dataset | IterableDataset: ...


def chat_dataset_from_harbor_job(
    path: str | Path | Iterable[str | Path],
    *,
    spec: ATIFSpec | None = None,
    reward_name: str | None = None,
    min_reward: float | None = None,
    streaming: bool = False,
) -> Dataset | IterableDataset:
    """Read saved full conversations or captured ATIF calls from completed Harbor jobs.

    Iterables select job directories, not individual trajectory files. Paths are
    captured once for replayable HF streaming; overlapping job paths are deduplicated.
    Full messages in trial metadata take precedence over ATIF trajectories.
    Summarized histories cannot reconstruct missing context. Missing grades stay
    null unless min_reward filters them out; ambiguous maps require reward_name.
    The optional spec configures ATIF extraction only.

    Args:
        path: Native artifact file, directory, or collection of source paths.
        spec: ATIF format spec used when a Harbor trial stores captured calls.
        reward_name: Verifier metric to select when a trial reports multiple rewards.
        min_reward: Explicit lower reward bound; omit it to retain failed attempts for RL.
        streaming: Return a lazy IterableDataset instead of materializing all rows.

    Returns:
        Complete Harbor chat rows with verifier grades and task checksums.
    """
    if min_reward is not None and not math.isfinite(min_reward):
        raise ValueError("min_reward must be finite")
    paths = [path] if isinstance(path, (str, Path)) else path
    jobs = tuple(sorted({Path(item).resolve() for item in paths}))
    if not jobs:
        raise ValueError("No Harbor job directories selected")
    return from_generator(
        partial(_chats_from_harbor_jobs, jobs, spec or ATIFSpec(), reward_name, min_reward),
        FEATURES,
        streaming=streaming,
    )


def _selected_reward(rewards: Mapping[str, Any], name: str | None) -> float | None:
    """Select one finite verifier reward, requiring a name when ambiguous."""
    if name is None:
        if "reward" in rewards:
            name = "reward"
        elif len(rewards) == 1:
            name = next(iter(rewards))
        elif rewards:
            raise ValueError("Multiple verifier rewards; specify reward_name")
        else:
            return None
    if name not in rewards:
        raise ValueError(f"Missing verifier reward: {name}")
    value = rewards[name]
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError(f"Verifier reward {name!r} must be finite and numeric")
    return float(value)


def _job_trials(job: Path) -> list[Path]:
    """Resolve completed trial directories from native Harbor jobs."""
    if not job.exists():
        raise FileNotFoundError(job)
    if not job.is_dir():
        raise ValueError(f"{job}: supply a Harbor job directory")
    summary = read_object(job / "result.json")
    if not summary.get("finished_at"):
        raise LookupError(f"{job}: Harbor job is unfinished")
    paths = sorted({*job.glob("*/result.json"), *job.glob("trials/*/result.json")})
    if len(paths) != summary.get("n_total_trials"):
        raise ValueError(f"{job}: missing or unexpected trial artifacts")
    return paths


def _trial_records(
    trial: Mapping[str, Any],
    path: Path,
    spec: ATIFSpec,
    reward: float | None,
    group: str,
) -> Iterator[dict[str, Any]]:
    """Read complete model inputs and outputs from a Harbor trial."""
    metadata = (trial.get("agent_result") or {}).get("metadata") or {}
    if "all_messages" in metadata:
        if metadata.get("summarization_count", 0) != 0:
            raise ValueError(f"{path}: saved message history must not be summarized")
        kwargs = ((trial.get("config") or {}).get("agent") or {}).get("kwargs") or {}
        request = kwargs.get("llm_call_kwargs") or {}
        configured = {**request, **(request.get("extra_body") or {})}.get("chat_template_kwargs")
        recorded = metadata.get("chat_template_kwargs")
        if recorded is not None and configured is not None and recorded != configured:
            raise ValueError(f"{path}: conflicting recorded chat-template settings")
        record = validate_record(
            {
                "messages": metadata["all_messages"],
                "tools": metadata.get("tools"),
                "chat_template_kwargs": recorded if recorded is not None else configured,
            }
        )
        yield {
            **record,
            "reward": reward,
            "group_id": group,
            "model_id": ((trial.get("config") or {}).get("agent") or {}).get("model_name"),
        }
        return
    trajectories = [path.parent / "agent/trajectory.json", path.parent / "trajectory.json"]
    found = [candidate for candidate in trajectories if candidate.is_file()]
    if len(found) != 1:
        raise ValueError(
            f"{path}: expected saved full messages or exactly one captured ATIF trajectory"
        )
    trajectory = read_object(found[0])
    for record in spec.iter_chats(trajectory, reward=reward, group_id=group):
        yield {**record, "session_id": trajectory.get("session_id")}


def _trial_sampling(trial: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the model identity and sampling settings of a Harbor trial."""
    kwargs = ((trial.get("config") or {}).get("agent") or {}).get("kwargs") or {}
    return sampling_parameters({**kwargs, **(kwargs.get("llm_call_kwargs") or {})})


def _trial_chats(
    path: Path,
    job: Path,
    spec: ATIFSpec,
    reward_name: str | None,
    min_reward: float | None,
) -> Iterator[ChatRow]:
    """Attach terminal trial grades and task identity to its model calls."""
    trial = read_object(path)
    if not trial.get("finished_at"):
        raise LookupError(f"{path}: Harbor trial is unfinished")
    if trial.get("exception_info") or trial.get("step_results"):
        raise ValueError(f"{path}: errored or multi-step trial requires explicit selection")
    identifier, group = trial.get("id"), trial.get("task_checksum")
    if not isinstance(identifier, str) or not identifier or not isinstance(group, str) or not group:
        raise ValueError(f"{path}: missing trial ID or task checksum")
    reward = _selected_reward((trial.get("verifier_result") or {}).get("rewards") or {}, reward_name)
    if min_reward is not None and (reward is None or reward < min_reward):
        return
    for record in _trial_records(trial, path, spec, reward, group):
        yield chat_row(
            {
                **record,
                "trace_id": identifier,
                "metadata_json": json.dumps(
                    {
                        "job_path": str(job),
                        "trial_path": str(path.parent),
                        "task_name": trial.get("task_name"),
                        "session_id": record.get("session_id"),
                        "sampling": _trial_sampling(trial),
                    },
                    allow_nan=False,
                ),
            }
        )


def _chats_from_harbor_jobs(
    jobs: tuple[Path, ...], spec: ATIFSpec, reward_name: str | None, min_reward: float | None
) -> Iterator[ChatRow]:
    """Yield replayable conversations from completed native Harbor jobs."""
    seen: set[str] = set()
    for job in jobs:
        for path in _job_trials(job):
            identifier = read_object(path).get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{path}: missing trial ID")
            if identifier in seen:
                raise ValueError(f"{path}: duplicate Harbor trial ID")
            seen.add(identifier)
            yield from _trial_chats(path, job, spec, reward_name, min_reward)
