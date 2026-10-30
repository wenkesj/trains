import json

import pytest

from codex_tracing.agent import _agent_result


def test_context_exhaustion_keeps_attempt_for_native_file_grading():
    stdout = json.dumps(
        {
            "type": "turn.failed",
            "error": {
                "message": (
                    "stream disconnected before completion: Requested token count exceeds the model's "
                    "maximum context length of 32768 tokens. You requested a total of 32786 tokens."
                )
            },
        }
    )
    result = _agent_result(1, stdout)
    assert result.stop_reason == "context_limit"
    assert result.response is None


def test_other_harness_failures_still_abort():
    stdout = json.dumps({"type": "turn.failed", "error": {"message": "Authentication failed"}})
    with pytest.raises(RuntimeError, match="codex exited"):
        _agent_result(1, stdout)


def test_completed_output_is_preserved():
    stdout = "upstream installer status\n" + json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "Implemented the solution."},
        }
    )
    result = _agent_result(0, stdout)
    assert result.stop_reason == "completed"
    assert result.response == "Implemented the solution."
