import pytest

from trains.datasets import chat_dataset_from_records
from trains.formats.records import validate_record


def test_invalid_chat_data_fails_before_training():
    with pytest.raises(ValueError, match="Orphan"):
        chat_dataset_from_records(
            [
                {
                    "messages": [
                        {"role": "tool", "tool_call_id": "missing", "content": "result"},
                        {"role": "assistant", "content": "done"},
                    ]
                }
            ]
        )


def test_missing_tool_results_are_rejected():
    with pytest.raises(ValueError, match="tool result"):
        validate_record(
            {
                "messages": [
                    {"role": "user", "content": "go"},
                    {"role": "tool", "tool_call_id": "missing", "content": "one"},
                    {"role": "assistant", "content": "done"},
                ]
            }
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"role": "assistant", "reasoning_content": 1, "content": "answer"},
        {"role": "assistant", "reasoning_content": "one", "reasoning": "two"},
        {"role": "assistant", "content": "answer", "weight": float("nan")},
        {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "not-json"}}]},
        {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "[]"}}]},
    ],
)
def test_invalid_reasoning_and_tool_payloads_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_record({"messages": [{"role": "user", "content": "go"}, bad]})


def test_parallel_tool_results_are_matched_by_id():
    calls = [
        {"id": name, "function": {"name": "lookup", "arguments": {"key": name}}}
        for name in ("a", "b")
    ]
    record = {
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": calls},
            {"role": "tool", "tool_call_id": "b", "content": "two"},
            {"role": "tool", "tool_call_id": "a", "content": "one"},
            {"role": "assistant", "content": "done"},
        ]
    }
    assert validate_record(record) == record
    calls[1]["id"] = "a"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_record(record)
