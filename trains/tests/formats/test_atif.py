import json

import pytest
from atif import Trajectory

from trains.formats.atif import ATIFSpec, _captured_generation


def test_atif_native_validation_and_exact_capture(processor, tmp_path):
    trajectory = {
        "schema_version": "ATIF-v1.8",
        "session_id": "trace-1",
        "agent": {"name": "agent", "version": "1", "model_name": "model"},
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "message": "north",
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
                                    "message": {"role": "assistant", "content": "north"},
                                }
                            ]
                        },
                    }
                },
            }
        ],
    }
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(trajectory))
    data = list(
        ATIFSpec(validation_class=Trajectory).iter_chats(trajectory, reward=1, group_id="task")
    )
    assert data[0]["trace_id"] == "trace-1" and data[0]["reward"] == 1
    assert ATIFSpec.from_trajectory(Trajectory.model_validate(trajectory)).to_chats()
    trajectory["steps"][0]["extra"] = {}
    path.write_text(json.dumps(trajectory))
    with pytest.raises(ValueError, match="exact model"):
        list(ATIFSpec().iter_chats(trajectory))


@pytest.mark.parametrize("nested_options", [False, True])
def test_exact_capture_preserves_tools_rendering_and_supervision(nested_options):
    tools = [{"type": "function", "function": {"name": "move", "parameters": {}}}]
    options = {"enable_thinking": False}
    request = {
        "model": "model",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {"role": "assistant", "content": "thinking"},
            {"role": "user", "content": "now"},
        ],
        "tools": tools,
        **(
            {"extra_body": {"chat_template_kwargs": options}}
            if nested_options
            else {"chat_template_kwargs": options}
        ),
    }
    output = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "move", "arguments": '{"direction":"north"}'},
            }
        ],
    }
    captured = {
        "request": request,
        "response": {"choices": [{"finish_reason": "tool_calls", "message": output}]},
    }
    (record,) = _captured_generation(captured, "model")
    assert record["tools"] == tools
    assert record["chat_template_kwargs"] == options
    assert record["messages"][0]["content"] == "go"
    assert record["messages"][1]["weight"] == 0
    assert record["messages"][-1] == output
    assert "weight" not in request["messages"][1]
    request.pop("tools")
    with pytest.raises(ValueError, match="actual tool schemas"):
        _captured_generation(captured, "model")
