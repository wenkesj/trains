import json

import pytest
from trains.datasets import chat_dataset_from_records

import source


def _row():
    return {
        "repo": "example/project", "license": "MIT", "dataset": "example/tasks",
        "instance_id": "task-1", "trajectory_id": "attempt-1",
        "trajectory": [
            {"role": "user", "content": "Read the project."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "read", "arguments": '{"path":"main.py"}'},
            }]},
            {"role": "tool", "content": "print('hello')"},
            {"role": "assistant", "content": "The project prints hello."},
        ],
    }


def test_stream_pins_revision_and_preserves_full_conversation_and_attribution(monkeypatch):
    row = _row()
    original = json.dumps(row)

    class Stream:
        def take(self, limit):
            assert limit == 1
            return [row]

    def load(name, *, revision, split, streaming):
        assert (name, revision, split, streaming) == (
            source.DATASET, source.REVISION, "train", True,
        )
        return Stream()

    monkeypatch.setattr(source, "load_dataset", load)
    chats = chat_dataset_from_records(source.load_conversations(1))
    messages = chats[0]["messages"]
    assert len(messages) == 4
    assert messages[2]["tool_call_id"] == "call-1"
    assert messages[1]["tool_calls"] == row["trajectory"][1]["tool_calls"]
    assert messages[-1]["content"] == row["trajectory"][-1]["content"]
    assert json.dumps(row) == original
    metadata = json.loads(chats[0]["metadata_json"])
    assert metadata["dataset_license"] == "CC-BY-4.0"
    assert metadata["repository_license"] == "MIT"
    assert metadata["attribution"] == source.ATTRIBUTION
    assert metadata["revision"] == source.REVISION
    assert chats[0]["group_id"] == "task-1"


def test_rejects_ambiguous_tool_results():
    row = _row()
    row["trajectory"][1]["tool_calls"].append({
        "id": "call-2", "type": "function",
        "function": {"name": "read", "arguments": '{"path":"other.py"}'},
    })
    with pytest.raises(ValueError, match="unambiguous"):
        source._conversation(row)


@pytest.mark.parametrize("license", ["", "unknown", "CC-BY-NC-4.0"])
def test_rejects_unexpected_repository_terms(license):
    row = _row()
    row["license"] = license
    with pytest.raises(ValueError, match="repository license"):
        source._conversation(row)
