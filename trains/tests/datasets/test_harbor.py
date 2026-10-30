import json

import pytest

from datasets import IterableDataset, load_from_disk
from trains.datasets import chat_dataset_from_harbor_job


def job(tmp_path, name="job", nested=False, rewards=None, trial_id="trial-1"):
    root = tmp_path / name
    trial = root / ("trials/trial" if nested else "trial")
    (trial / "agent").mkdir(parents=True)
    (root / "result.json").write_text(
        json.dumps({"finished_at": "2026-01-01", "n_total_trials": 1})
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "id": trial_id,
                "task_name": "task",
                "task_checksum": "task-v1",
                "finished_at": "2026-01-01",
                "verifier_result": {
                    "rewards": rewards if rewards is not None else {"reward": 0.75}
                },
            }
        )
    )
    trajectory = {
        "schema_version": "ATIF-v1.8",
        "session_id": "session",
        "agent": {"model_name": "model"},
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "message": "done",
                "extra": {
                    "trains": {
                        "request": {
                            "model": "model",
                            "messages": [{"role": "user", "content": "go"}],
                        },
                        "response": {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": "done"},
                                }
                            ]
                        },
                    }
                },
            }
        ],
    }
    (trial / "agent/trajectory.json").write_text(json.dumps(trajectory))
    return root, trial


@pytest.mark.parametrize("nested", [False, True])
def test_native_artifacts_supply_identity_grade_and_group(tmp_path, nested):
    root, _ = job(tmp_path, nested=nested)
    dataset = chat_dataset_from_harbor_job(str(root))
    row = dataset[0]
    assert row["trace_id"] == "trial-1" and row["group_id"] == "task-v1"
    assert row["reward"] == 0.75 and row["model_id"] == "model"
    assert json.loads(row["metadata_json"])["session_id"] == "session"
    dataset.save_to_disk(tmp_path / "saved")
    assert load_from_disk(tmp_path / "saved")[:] == dataset[:]


def test_multiple_jobs_and_overlap_are_replayable(tmp_path):
    first, _ = job(tmp_path, "a", trial_id="one")
    second, _ = job(tmp_path, "b", trial_id="two")
    stream = chat_dataset_from_harbor_job(iter([second, first, first]), streaming=True)
    assert isinstance(stream, IterableDataset)
    assert [row["trace_id"] for row in stream] == ["one", "two"]
    assert list(stream) == list(stream)


def test_missing_grade_is_null_and_reward_selection_is_explicit(tmp_path):
    root, trial = job(tmp_path, rewards={})
    assert next(iter(chat_dataset_from_harbor_job(root, streaming=True)))["reward"] is None
    result = json.loads((trial / "result.json").read_text())
    result["verifier_result"]["rewards"] = {"accuracy": 1, "style": 0}
    (trial / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Multiple verifier"):
        list(chat_dataset_from_harbor_job(root, streaming=True))
    assert (
        next(iter(chat_dataset_from_harbor_job(root, reward_name="style", streaming=True)))[
            "reward"
        ]
        == 0
    )


def test_trial_id_collisions_and_unfinished_jobs_fail(tmp_path):
    first, _ = job(tmp_path, "a")
    second, _ = job(tmp_path, "b")
    with pytest.raises(ValueError, match="duplicate"):
        list(chat_dataset_from_harbor_job([first, second], streaming=True))
    (first / "result.json").write_text(json.dumps({"n_total_trials": 1}))
    with pytest.raises(LookupError, match="unfinished"):
        list(chat_dataset_from_harbor_job(first, streaming=True))


def test_glob_selects_job_directories(tmp_path):
    job(tmp_path, "run-a", trial_id="one")
    job(tmp_path, "run-b", trial_id="two")
    assert len(chat_dataset_from_harbor_job(tmp_path.glob("run-*"))) == 2
    with pytest.raises(ValueError, match="No Harbor"):
        chat_dataset_from_harbor_job([])


def native_job(tmp_path, name="native", *, metadata=None, trial_id="native-trial", rewards=None):
    root, trial = job(tmp_path, name, trial_id=trial_id, rewards=rewards)
    result = json.loads((trial / "result.json").read_text())
    result["agent_result"] = {
        "metadata": metadata
        if metadata is not None
        else {
            "all_messages": [
                {"role": "user", "content": "full context"},
                {"role": "assistant", "content": "native answer", "reasoning_content": "plan"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ],
            "chat_template_kwargs": {"enable_thinking": True},
        }
    }
    (trial / "result.json").write_text(json.dumps(result))
    return root, trial


def test_full_history_is_agent_independent_and_takes_precedence_over_atif(tmp_path):
    root, _ = native_job(tmp_path)
    rows = list(chat_dataset_from_harbor_job(root, streaming=True))
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"] == "native answer"
    assert rows[0]["messages"][-1]["reasoning_content"] == "plan"
    assert rows[0]["tools"][0]["function"]["name"] == "lookup"
    assert rows[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert rows[0]["trace_id"] == "native-trial"
    assert rows[0]["group_id"] == "task-v1"
    assert json.loads(rows[0]["metadata_json"])["trial_path"].endswith("trial")


def test_terminus_native_metadata_retains_configured_request_template(tmp_path):
    root, trial = native_job(
        tmp_path,
        metadata={
            "all_messages": [
                {"role": "user", "content": "full context"},
                {"role": "assistant", "content": "native answer"},
            ],
            "summarization_count": 0,
        },
    )
    result = json.loads((trial / "result.json").read_text())
    options = {"enable_thinking": False, "preserve_thinking": True}
    result["config"] = {
        "agent": {
            "model_name": "openai/liquid-sampler",
            "kwargs": {
                "temperature": 1,
                "llm_call_kwargs": {
                    "top_p": 1,
                    "extra_body": {
                        "top_k": -1,
                        "min_p": 0,
                        "chat_template_kwargs": options,
                    },
                },
            },
        }
    }
    (trial / "result.json").write_text(json.dumps(result))
    (row,) = list(chat_dataset_from_harbor_job(root, streaming=True))
    assert row["chat_template_kwargs"] == options
    assert row["model_id"] == "openai/liquid-sampler"
    assert json.loads(row["metadata_json"])["sampling"] == {
        "temperature": 1,
        "top_p": 1,
        "top_k": -1,
        "min_p": 0,
    }
    result["agent_result"]["metadata"]["chat_template_kwargs"] = {"enable_thinking": True}
    (trial / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="conflicting recorded"):
        list(chat_dataset_from_harbor_job(root, streaming=True))


def test_one_importer_handles_mixed_job_artifacts_and_replays(tmp_path):
    native, _ = native_job(tmp_path)
    atif, _ = job(tmp_path, trial_id="atif-trial")
    stream = chat_dataset_from_harbor_job([native, atif], streaming=True)
    rows = list(stream)
    assert {row["trace_id"] for row in rows} == {"native-trial", "atif-trial"}
    assert list(stream) == rows
    assert len(chat_dataset_from_harbor_job([native, atif])) == 2


@pytest.mark.parametrize("native", [False, True])
def test_reward_filter_is_optional_for_every_artifact_path(tmp_path, native):
    root, _ = native_job(tmp_path) if native else job(tmp_path)
    assert len(list(chat_dataset_from_harbor_job(root, streaming=True))) == 1
    assert list(chat_dataset_from_harbor_job(root, min_reward=1, streaming=True)) == []
    assert len(chat_dataset_from_harbor_job(root, min_reward=1)) == 0
    assert len(chat_dataset_from_harbor_job(root, min_reward=0.5)) == 1
    missing, _ = (
        native_job(tmp_path, "missing", rewards={})
        if native
        else job(tmp_path, "missing", rewards={})
    )
    assert next(iter(chat_dataset_from_harbor_job(missing, streaming=True)))["reward"] is None
    assert list(chat_dataset_from_harbor_job(missing, min_reward=0, streaming=True)) == []


def test_summarized_or_invalid_full_history_does_not_fall_back_silently(tmp_path):
    root, _ = native_job(tmp_path, metadata={"all_messages": [], "summarization_count": 1})
    with pytest.raises(ValueError, match="summarized"):
        list(chat_dataset_from_harbor_job(root, streaming=True))
    root, _ = native_job(tmp_path, "invalid", metadata={"all_messages": []})
    with pytest.raises(ValueError):
        list(chat_dataset_from_harbor_job(root, streaming=True))


def test_missing_full_history_uses_atif_and_missing_both_fails(tmp_path):
    root, trial = native_job(tmp_path, metadata={"other_agent_field": "value"})
    assert (
        next(iter(chat_dataset_from_harbor_job(root, streaming=True)))["messages"][-1]["content"]
        == "done"
    )
    (trial / "agent/trajectory.json").unlink()
    with pytest.raises(ValueError, match="saved full messages or exactly one captured ATIF"):
        list(chat_dataset_from_harbor_job(root, streaming=True))


def test_sampling_provenance_excludes_agent_credentials(tmp_path):
    root, trial = job(tmp_path)
    path = trial / "result.json"
    result = json.loads(path.read_text())
    result["config"] = {
        "agent": {
            "kwargs": {"temperature": 1, "api_key": "secret", "llm_call_kwargs": {"top_p": 1}}
        }
    }
    path.write_text(json.dumps(result))
    metadata = json.loads(chat_dataset_from_harbor_job(root)[0]["metadata_json"])
    assert metadata["sampling"] == {"temperature": 1, "top_p": 1, "top_k": None, "min_p": None}
    assert "secret" not in json.dumps(metadata)
