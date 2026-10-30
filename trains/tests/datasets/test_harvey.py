import json
from types import SimpleNamespace

import pytest

pytest.importorskip("lab_core")

from lab_core.harness import run as harness
from lab_core.harness.adapters.baseten import BasetenAdapter
from lab_core.harness.agent_loop import run_agent
from openai.types.chat import ChatCompletion

from trains.datasets import chat_dataset_from_harvey_run
from trains.datasets.harvey import _harvey_record


def native_run(tmp_path, monkeypatch, *, termination=None):
    monkeypatch.setenv("BASETEN_API_KEY", "local")
    monkeypatch.setattr(harness, "load_task", lambda name: {"instructions": "Write the answer"})
    adapter = BasetenAdapter("test-model")
    answers = iter(
        [
            {
                "role": "assistant",
                "content": "working",
                "reasoning_content": "not persisted by LAB",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        ]
    )

    def complete(**kwargs):
        nonlocal calls
        calls += 1
        if termination == "context_overflow" and calls == 2:
            raise ValueError("context_length_exceeded")
        return ChatCompletion.model_validate(
            {
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "message": next(answers), "finish_reason": "stop"}],
            }
        )

    calls = 0
    adapter.client.chat.completions.create = complete
    executor = SimpleNamespace(execute=lambda *args: "/workspace", get_metrics=lambda: {})
    result = run_agent(
        adapter=adapter,
        system_prompt=harness.build_system_preamble(False),
        user_prompt="Write the answer",
        tool_executor=executor,
        transcript_path=str(tmp_path / "transcript.jsonl"),
        max_turns=1 if termination == "max_turns_exceeded" else 2,
    )
    identity = {"run_id": "run-1", "task": "area/task", "model": "baseten/test-model"}
    (tmp_path / "config.json").write_text(
        json.dumps({**identity, "enable_finish": False, "skills": [], "reasoning_effort": None})
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                **identity,
                "completed_at": "done",
                "finished_cleanly": result["finished_cleanly"],
                "turn_count": result["turn_count"],
                "finish_reason": result["finish_reason"],
                "context_overflow": result["context_overflow"],
                "max_turns_exceeded": result["max_turns_exceeded"],
            }
        )
    )
    (tmp_path / "scores.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "task": "area/task",
                "n_criteria": 2,
                "n_passed": 1,
                "n_grading_errors": 0,
                "all_pass": False,
            }
        )
    )
    return result


def test_native_baseten_transcript_rebuilds_context_without_changing_harness(tmp_path, monkeypatch):
    original = native_run(tmp_path, monkeypatch)
    record = _harvey_record(tmp_path, None, "criterion_pass")
    assert record["messages"][0]["content"] == harness.build_system_preamble(False)
    assert record["messages"][1]["content"] == "Write the answer"
    assert record["messages"][2]["tool_calls"][0]["function"]["name"] == "bash"
    assert record["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "/workspace",
    }
    assert record["messages"][-1]["content"] == "done"
    assert original["messages"][2]["reasoning_content"] == "not persisted by LAB"
    assert "reasoning_content" not in record["messages"][2]
    assert record["tools"][0]["type"] == "function"
    assert record["reward"] == 0.5
    assert "score" not in record["messages"][0]["content"]
    dataset = chat_dataset_from_harvey_run(tmp_path, streaming=True)
    assert next(iter(dataset))["trace_id"] == "run-1"
    assert _harvey_record(tmp_path, None, "all_pass")["reward"] == 0


def test_truncated_transcripts_and_wrong_scores_fail(tmp_path, monkeypatch):
    native_run(tmp_path, monkeypatch)
    (tmp_path / "transcript.jsonl").write_text("")
    with pytest.raises(ValueError, match="turn counts"):
        _harvey_record(tmp_path, None, "criterion_pass")


def test_importer_uses_saved_skill_and_finish_settings(tmp_path, monkeypatch):
    native_run(tmp_path, monkeypatch)
    skills = tmp_path / "skills"
    (skills / "fixture").mkdir(parents=True)
    (skills / "fixture/SKILL.md").write_text("Use careful analysis")
    monkeypatch.setattr(harness, "SKILLS_DIR", skills)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config.update(skills=["fixture"], enable_finish=True)
    config_path.write_text(json.dumps(config))
    record = _harvey_record(tmp_path, None, "criterion_pass")
    expected = harness.build_system_preamble(True) + harness.load_skills(["fixture"])
    assert record["messages"][0]["content"] == expected
    assert "finish" in {tool["function"]["name"] for tool in record["tools"]}
    (skills / "fixture/SKILL.md").unlink()
    with pytest.raises(ValueError, match="Missing original LAB skills"):
        _harvey_record(tmp_path, None, "criterion_pass")


@pytest.mark.parametrize("termination", ["max_turns_exceeded", "context_overflow"])
def test_completed_native_failures_retain_every_captured_action(tmp_path, monkeypatch, termination):
    result = native_run(tmp_path, monkeypatch, termination=termination)
    assert result["finished_cleanly"] is False
    assert result["finish_reason"] == termination
    record = _harvey_record(tmp_path, None, "criterion_pass")
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant", "tool"]
    assert record["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"command":"pwd"}'
    assert record["reward"] == 0.5
    assert json.loads(record["metadata_json"])["finish_reason"] == termination
    (tmp_path / "metrics.json").unlink()
    with pytest.raises(FileNotFoundError):
        _harvey_record(tmp_path, None, "criterion_pass")


def test_unknown_unclean_termination_is_not_treated_as_complete(tmp_path, monkeypatch):
    native_run(tmp_path, monkeypatch)
    path = tmp_path / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics.update(finished_cleanly=False, finish_reason="connection_lost")
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="recognized terminal reason"):
        _harvey_record(tmp_path, None, "criterion_pass")
