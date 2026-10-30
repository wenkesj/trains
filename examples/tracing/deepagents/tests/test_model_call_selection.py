from typing import Any

import pytest
from trains.formats.tracing import TraceRecord

from deepagents_tracing.spec import ModelCallSpec


def test_empty_failed_request_is_not_a_training_target_but_zero_reward_is_retained():
    error_attributes: dict[str, Any] = {"openinference.span.kind": "LLM"}
    error = {
        "SpanId": "failed",
        "SpanName": "ChatCompletion",
        "status_code": "ERROR",
        "SpanAttributes": error_attributes,
    }
    success = {
        "SpanId": "completed",
        "SpanName": "ChatCompletion",
        "status_code": "OK",
        "SpanAttributes": {
            "openinference.span.kind": "LLM",
            "input.value": {"messages": [{"role": "user", "content": "Implement it"}]},
            "output.value": {"role": "assistant", "content": "Incorrect solution"},
        },
    }
    trace = TraceRecord(id="attempt", spans=[error, success], metadata={"reward": 0})
    (row,) = ModelCallSpec().iter_chats(trace)
    assert row["span_id"] == "completed"
    assert row["reward"] == 0
    error_attributes["output.value"] = {"role": "assistant", "content": "Partial"}
    with pytest.raises(ValueError, match="partial model output"):
        list(ModelCallSpec().iter_chats(trace))


def test_explicitly_completed_error_response_keeps_generated_tool_action():
    attrs = {
        "openinference.span.kind": "LLM",
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "Implement it",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call-1",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "read_file",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"path":"solution.py"}',
        "input.value": {
            "messages": [{"role": "user", "content": "Implement it"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "execute",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        "output.value": {
            "object": "response",
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "output": [
                {
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call-1",
                    "name": "read_file",
                    "arguments": '{"path":"solution.py"}',
                }
            ],
        },
    }
    trace = TraceRecord(
        id="attempt",
        metadata={"reward": 0},
        spans=[
            {
                "SpanId": "completed",
                "SpanName": "Response",
                "status_code": "ERROR",
                "SpanAttributes": attrs,
            }
        ],
    )
    (row,) = ModelCallSpec().iter_chats(trace)
    assert row["reward"] == 0
    calls = row["messages"][-1]["tool_calls"]
    assert calls is not None
    assert calls[0]["function"]["name"] == "read_file"
