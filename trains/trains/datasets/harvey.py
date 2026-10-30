import json
import math
from collections.abc import Iterable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset, IterableDataset
from trains.datasets.chat import FEATURES, chat_row
from trains.datasets.generator import from_generator
from trains.datasets.harbor import read_object
from trains.formats.records import validate_record
from trains.formats.sampling import sampling_parameters
from trains.types import ChatRow

try:
    from lab_core.harness import run as _harness
    from lab_core.harness import tools as _native_tools
except ImportError as error:
    _harness = None
    _native_tools = None
    _HARVEY_IMPORT_ERROR = error
else:
    _HARVEY_IMPORT_ERROR = None


def chat_dataset_from_harvey_run(
    path: str | Path | Iterable[str | Path],
    *,
    score_file: str | None = None,
    reward_metric: Literal["criterion_pass", "all_pass"] = "criterion_pass",
    min_reward: float | None = None,
    streaming: bool = False,
) -> Dataset | IterableDataset:
    """Read native LAB runs, rebuilding model context with the optional lab-core library.

    Install trains-sdk[harvey] and configure LAB_ROOT for the original task data.
    Prompts, skills and tool schemas are loaded through lab-core's native helpers.
    Native transcript text and tool calls/results are retained in one
    full conversation. Reasoning absent from the transcript cannot be recovered.
    Optional native judge scores attach rewards without entering model inputs.
    Completed attempts that exhaust turns or context retain their captured actions
    and native grades; interrupted runs without completion metrics are rejected.

    Args:
        path: Native artifact file, directory, or collection of source paths.
        score_file: Native LAB score filename to select when multiple judge outputs exist.
        reward_metric: Native criterion-pass fraction or all-pass reward selection.
        min_reward: Explicit lower reward bound; omit it to retain failed attempts for RL.
        streaming: Return a lazy IterableDataset instead of materializing all rows.

    Returns:
        Complete LAB conversations with native run identities and selected judge grades.
    """
    if _harness is None or _native_tools is None:
        raise ImportError(
            "Harvey LAB support requires lab-core. Install it with pip install 'trains-sdk[harvey]'."
        ) from _HARVEY_IMPORT_ERROR
    paths = [path] if isinstance(path, (str, Path)) else path
    paths = tuple(sorted({Path(item).resolve() for item in paths}))
    if not paths:
        raise ValueError("No Harvey LAB runs selected")
    if reward_metric not in ("criterion_pass", "all_pass"):
        raise ValueError("Unsupported LAB reward metric")
    return from_generator(
        partial(_harvey_chats, paths, score_file, reward_metric, min_reward),
        FEATURES,
        streaming=streaming,
    )


def _conversation_from_transcript(
    path: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct the original LAB system prompt, task, skills, and tools."""
    if _harness is None or _native_tools is None:
        raise ImportError(
            "Harvey LAB support requires lab-core. Install it with pip install 'trains-sdk[harvey]'."
        ) from _HARVEY_IMPORT_ERROR
    enabled = config["enable_finish"]
    system = _harness.build_system_preamble(enabled)
    skills = config["skills"]
    if skills:
        missing = [
            name for name in skills if not (_harness.SKILLS_DIR / name / "SKILL.md").is_file()
        ]
        if missing:
            raise ValueError(f"Missing original LAB skills: {missing}")
        system += _harness.load_skills(skills)
    task = _harness.load_task(config["task"])
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task["instructions"]},
    ]
    # LAB increments the counter before a request that exhausts context; that
    # rejected request has no assistant action in the transcript.
    turns = metrics["turn_count"] - (metrics.get("finish_reason") == "context_overflow")
    messages.extend(_transcript_messages(path, turns))
    return {
        "messages": messages,
        "tools": [
            {"type": "function", "function": tool}
            for tool in _native_tools.get_all_tool_definitions(enable_finish=enabled)
        ],
    }


def _transcript_messages(path: Path, expected_turns: int) -> list[dict[str, Any]]:
    """Read complete LAB turns and require the recorded turn count."""
    messages = []
    turns = 0
    for line in (path / "transcript.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry["role"] == "assistant":
            turns += 1
            if entry["turn"] != turns:
                raise ValueError("LAB transcript assistant turns are missing or out of order")
            message = {"role": "assistant", "content": entry["text"]}
            if entry.get("tool_calls"):
                message["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                    for call in entry["tool_calls"]
                ]
            if entry.get("reasoning_content"):
                message["reasoning_content"] = entry["reasoning_content"]
            messages.append(message)
        elif entry["role"] == "tool" and entry["turn"] == turns:
            messages.append(
                {"role": "tool", "tool_call_id": entry["tool_call_id"], "content": entry["result"]}
            )
        else:
            raise ValueError("Unexpected or out-of-order LAB transcript entry")
    if not turns or turns != expected_turns:
        raise ValueError("LAB transcript and metrics turn counts must match")
    return messages


def _harvey_record(
    path: Path, score_file: str | None, reward_metric: Literal["criterion_pass", "all_pass"]
) -> dict[str, Any]:
    """Combine a complete LAB transcript with its validated run settings."""
    config = read_object(path / "config.json")
    metrics = read_object(path / "metrics.json")
    terminal_failure = metrics.get("finished_cleanly") is False and metrics.get(
        "finish_reason"
    ) in {"max_turns_exceeded", "context_overflow"}
    if not metrics.get("completed_at") or not (metrics.get("finished_cleanly") or terminal_failure):
        raise ValueError(f"{path}: select a completed LAB run with a recognized terminal reason")
    if any(
        not config.get(key) or config[key] != metrics.get(key)
        for key in ("run_id", "task", "model")
    ):
        raise ValueError(f"{path}: config and metrics identities must match")
    if not config["model"].startswith("baseten/"):
        raise ValueError("This importer currently supports LAB's Baseten Chat Completions adapter")
    record = _conversation_from_transcript(path, config, metrics)
    effort = config.get("reasoning_effort")
    if effort and effort.lower() != "none":
        record["chat_template_kwargs"] = {"enable_thinking": True}
    record = validate_record(record)
    reward = _harvey_reward(path, config, score_file, reward_metric)
    return {
        **record,
        "reward": reward,
        "model_id": config["model"],
        "trace_id": config["run_id"],
        "group_id": config["task"],
        "metadata_json": json.dumps(
            {
                "model": config["model"],
                "context_source": "lab_core",
                "finished_cleanly": metrics["finished_cleanly"],
                "finish_reason": metrics.get("finish_reason"),
                "sampling": sampling_parameters(config),
                "reasoning_capture": "only_if_present_in_transcript",
            }
        ),
    }


def _harvey_reward(
    path: Path,
    config: dict[str, Any],
    score_file: str | None,
    reward_metric: Literal["criterion_pass", "all_pass"],
) -> float | None:
    """Read the selected native LAB judge score and reject grading errors."""
    score = None
    if score_file:
        score = read_object(path / score_file)
    else:
        candidates = [
            file for file in (path / "scores.json", path / "scores_dual.json") if file.is_file()
        ]
        if len(candidates) > 1:
            raise ValueError("Select score_file when both single- and dual-judge scores exist")
        if candidates:
            score = read_object(candidates[0])
    reward = None
    if score is not None:
        if score.get("run_id") != config["run_id"] or score.get("task") != config["task"]:
            raise ValueError("LAB scores do not belong to this run/task")
        judges = list((score.get("per_judge") or {}).values()) or [score]
        if any(judge.get("n_grading_errors", 0) for judge in judges):
            raise ValueError("LAB judge errors must be resolved before training")
        if reward_metric == "all_pass":
            if type(score.get("all_pass")) is not bool:
                raise ValueError("LAB score is missing all_pass")
            reward = float(score["all_pass"])
        elif "dual_criterion_pass" in score:
            reward = score["dual_criterion_pass"]
        else:
            if not score.get("n_criteria"):
                raise ValueError("LAB score is missing criterion counts")
            reward = score["n_passed"] / score["n_criteria"]
        if type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("LAB reward must be finite and between zero and one")
    return reward


def _harvey_chats(
    paths: tuple[Path, ...],
    score_file: str | None,
    reward_metric: Literal["criterion_pass", "all_pass"],
    min_reward: float | None,
) -> Iterator[ChatRow]:
    """Yield full LAB conversations with task identities and terminal grades."""
    for path in paths:
        record = _harvey_record(path, score_file, reward_metric)
        reward = record["reward"]
        if min_reward is None or (reward is not None and reward >= min_reward):
            yield chat_row(record)
