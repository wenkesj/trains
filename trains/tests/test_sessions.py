"""Native session conversion regression tests."""

import json

import pytest

from datasets import Dataset, Features, Json, Value
from trains.datasets import chat_dataset_from_sessions
from trains.sessions.adapters import (
    CodexSessionAdapter,
    OpenCodeSessionAdapter,
    PiSessionAdapter,
    STSSessionAdapter,
)


def write_events(path, events):
    path.write_text("\n".join(json.dumps(event) for event in events))
    return path


def test_codex_parallel_tools_and_duplicate_ui_events(tmp_path):
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "test"}]},
        *[
            {"type": "function_call", "call_id": key, "name": "run", "arguments": "{}"}
            for key in ["a", "b"]
        ],
        *[{"type": "function_call_output", "call_id": key, "output": "ok"} for key in ["a", "b"]],
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "done"}],
        },
    ]
    events = [{"type": "session_meta", "payload": {"id": "session-1"}}]
    events += [{"type": "response_item", "payload": item} for item in items]
    events += [{"type": "event_msg", "payload": {"type": "agent_message", "message": "done"}}]
    path = write_events(tmp_path / "rollout.jsonl", events)
    row = chat_dataset_from_sessions(path, adapter=CodexSessionAdapter())[0]
    assert row["trace_id"] == "session-1"
    assert len(row["messages"]) == 5
    assert [call["id"] for call in row["messages"][1]["tool_calls"]] == ["a", "b"]


def test_sts_aliases_and_orphan_rejection(tmp_path):
    path = write_events(
        tmp_path / "session.jsonl",
        [
            {"type": "session", "id": "sts-1", "harness": "example"},
            {
                "type": "message",
                "message": {"role": "assistant", "content": "ok", "reasoningContent": "think"},
            },
        ],
    )
    assert (
        chat_dataset_from_sessions(path, adapter=STSSessionAdapter())[0]["messages"][0][
            "reasoning_content"
        ]
        == "think"
    )
    write_events(
        path,
        [
            {
                "type": "message",
                "message": {"role": "tool", "content": "bad", "toolCallId": "missing"},
            },
            {"type": "message", "message": {"role": "assistant", "content": "done"}},
        ],
    )
    with pytest.raises(ValueError, match=r"session\.jsonl.*Orphan"):
        chat_dataset_from_sessions(path, adapter=STSSessionAdapter())


def test_opencode_export(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(
        json.dumps(
            {
                "info": {"id": "oc-1"},
                "messages": [
                    {"info": {"role": "user"}, "parts": [{"type": "text", "text": "test"}]},
                    {
                        "info": {"role": "assistant"},
                        "parts": [
                            {
                                "type": "tool",
                                "tool": "bash",
                                "callID": "call-1",
                                "state": {
                                    "status": "completed",
                                    "input": {"command": "true"},
                                    "output": "",
                                },
                            },
                        ],
                    },
                    {"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "done"}]},
                ],
            }
        )
    )
    row = chat_dataset_from_sessions(path, adapter=OpenCodeSessionAdapter())[0]
    assert row["messages"][2]["tool_call_id"] == "call-1"
    assert row["trace_id"] == "oc-1"


def test_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="No sessions matching"):
        chat_dataset_from_sessions(tmp_path, adapter=CodexSessionAdapter())


def test_pi_tool_calls_and_reasoning(tmp_path):
    path = write_events(
        tmp_path / "pi.jsonl",
        [
            {"type": "session", "id": "pi-1"},
            {
                "type": "message",
                "message": {"role": "user", "content": [{"type": "text", "text": "run"}]},
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "check"},
                        {
                            "type": "toolCall",
                            "id": "one",
                            "name": "bash",
                            "arguments": {"command": "true"},
                        },
                    ],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "one",
                    "content": [{"type": "text", "text": "ok"}],
                },
            },
            {
                "type": "message",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            },
        ],
    )
    row = chat_dataset_from_sessions(path, adapter=PiSessionAdapter())[0]
    assert row["messages"][1]["reasoning_content"] == "check"
    assert row["messages"][2] == {"role": "tool", "content": "ok", "tool_call_id": "one"}


def test_custom_adapter_and_recursive_discovery(tmp_path):
    class CustomAdapter:
        pattern = "*.trace"

        def read(self, path):
            return self.to_chat(path.read_text())

        def to_chat(self, session):
            return {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": session},
                ],
                "metadata_json": json.dumps({"custom": True}),
            }

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "one.trace").write_text("answer")
    (nested / "ignored.jsonl").write_text("not an event")
    row = chat_dataset_from_sessions(tmp_path, adapter=CustomAdapter())[0]
    assert row["messages"][-1]["content"] == "answer"
    assert json.loads(row["metadata_json"]) == {"custom": True, "source": str(nested / "one.trace")}


def test_explicit_adapter_method_in_hf_map():
    session = "\n".join(
        json.dumps(event)
        for event in [
            {"type": "session", "id": "mapped-session"},
            {"type": "message", "message": {"role": "user", "content": "Hi"}},
            {"type": "message", "message": {"role": "assistant", "content": "Hello"}},
        ]
    )
    dataset = Dataset.from_dict({"session": [session, session]})
    adapter = STSSessionAdapter()
    chats = dataset.map(
        lambda row: adapter.to_chat(row["session"]),
        remove_columns=dataset.column_names,
        features=Features({"messages": Json(), "trace_id": Value("string")}),
    )
    assert len(chats) == 2
    assert chats[1]["messages"][-1]["content"] == "Hello"
    assert chats[0]["trace_id"] == "mapped-session"
    assert not callable(adapter)


def test_pi_empty_request_error_keeps_failed_attempt_and_sampled_actions(tmp_path):
    events = [
        {"type": "session", "id": "failed-attempt"},
        {"type": "message", "message": {"role": "user", "content": "Implement it"}},
        {"type": "message", "message": {"role": "assistant", "content": "Trying."}},
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [],
                "stopReason": "error",
                "errorMessage": "model-call budget exhausted",
                "usage": {"output": 0},
            },
        },
    ]
    path = write_events(tmp_path / "failed.jsonl", events)
    row = chat_dataset_from_sessions(path, adapter=PiSessionAdapter())[0]
    assert row["trace_id"] == "failed-attempt"
    assert [m["content"] for m in row["messages"]] == ["Implement it", "Trying."]
    events[-1]["message"]["usage"]["output"] = 10
    write_events(path, events)
    with pytest.raises(ValueError, match="generated tokens without captured content"):
        chat_dataset_from_sessions(path, adapter=PiSessionAdapter())
