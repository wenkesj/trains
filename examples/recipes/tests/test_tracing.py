import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Status, StatusCode
from recipes.tracing import (
    ModelCallRecorder,
    canonical_tool_schemas,
    has_completed_output,
    langsmith_run_id,
)


def test_tool_schema_order_survives_provider_json_reordering_without_mutating_input():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workdir": {"type": "string", "description": "Directory"},
                        "command": {"type": "string", "description": "Command"},
                    },
                    "required": ["workdir", "command"],
                },
            },
        }
    ]
    original = json.dumps(tools)
    canonical = canonical_tool_schemas(tools)
    provider_copy = json.loads(json.dumps(canonical, sort_keys=True))
    assert json.dumps(canonical) == json.dumps(provider_copy)
    assert canonical == tools
    assert canonical[0]["function"]["parameters"]["required"] == ["workdir", "command"]
    assert json.dumps(tools) == original
    assert canonical[0] is not tools[0]


def test_receipt_records_only_completed_sdk_calls_and_releases_trace_state():
    recorder = ModelCallRecorder({"messages.create", "ChatCompletion"})
    provider = TracerProvider()
    provider.add_span_processor(recorder)
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("agent.summary") as root:
        trace_id = f"{root.get_span_context().trace_id:032x}"
        with tracer.start_as_current_span("messages.create") as first:
            first_id = f"{first.get_span_context().span_id:016x}"
            first.set_attribute("input.value", "must not be retained")
        with tracer.start_as_current_span("ChatCompletion") as second:
            second_id = f"{second.get_span_context().span_id:016x}"
        with tracer.start_as_current_span("messages.create") as failed:
            failed.set_status(Status(StatusCode.ERROR, "request failed before generation"))
    assert recorder.take(trace_id) == sorted([first_id, second_id])
    assert recorder.take(trace_id) == []
    provider.shutdown()


def test_completed_generation_is_kept_despite_error_span_status():
    output = {
        "object": "response",
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "output": [
            {
                "type": "function_call",
                "status": "completed",
                "name": "read_file",
                "arguments": '{"path":"solution.py"}',
            }
        ],
    }
    attributes = {"output.value": json.dumps(output)}
    assert has_completed_output(attributes)
    recorder = ModelCallRecorder({"Response"})
    provider = TracerProvider()
    provider.add_span_processor(recorder)
    with provider.get_tracer(__name__).start_as_current_span("Response") as span:
        span.set_attribute("output.value", attributes["output.value"])
        span.set_status(Status(StatusCode.ERROR, "tool failed after generation"))
        trace_id = f"{span.get_span_context().trace_id:032x}"
        span_id = f"{span.get_span_context().span_id:016x}"
    assert recorder.take(trace_id) == [span_id]
    for changes in [
        {"status": "incomplete"},
        {"error": {"code": "failed"}},
        {"incomplete_details": {"reason": "max_output_tokens"}},
        {"output": [{"status": "in_progress"}]},
    ]:
        assert not has_completed_output({"output.value": {**output, **changes}})
    provider.shutdown()


def test_langsmith_native_id_mapping_preserves_every_span_bit():
    assert langsmith_run_id("ede683fce5850b83") == "00000000-0000-0000-ede6-83fce5850b83"
    assert langsmith_run_id("0000000000000001") == "00000000-0000-0000-0000-000000000001"
    with pytest.raises(ValueError, match="OpenTelemetry"):
        langsmith_run_id("sdk-a")
