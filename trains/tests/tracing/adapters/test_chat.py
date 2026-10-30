import json

import pandas as pd
import pytest

from trains.formats.tracing import TraceRecord, TraceSpec
from trains.tracing.adapters.registry import registry
from trains.tracing.providers.utils import frame_records, traces_from_spans


def attributes():
    return {
        "openinference.span.kind": "LLM",
        "llm.model_name": "policy",
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "go",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "north",
    }


def test_openinference_needs_no_custom_markers_and_uses_per_span_model():
    trace = TraceRecord(id="trace", spans=[{"SpanId": "call", "SpanAttributes": attributes()}])
    (record,) = TraceSpec().iter_chats(trace)
    assert record["messages"][-1]["content"] == "north"
    assert record["model_id"] == "policy"
    assert record["span_id"] == "call"


def test_standard_fields_take_precedence_over_display_summaries():
    attrs = {**attributes(), "input.value": "summary", "output.value": "summary"}
    record = registry.convert([{"SpanAttributes": attrs}])[0][0]
    assert record["messages"][0]["content"] == "go"
    assert record["messages"][-1]["content"] == "north"


def test_tool_only_output_and_empty_schema_preserve_call_boundaries():
    attrs = attributes()
    attrs.pop("llm.output_messages.0.message.content")
    attrs.update(
        {
            "llm.input_messages.1.message.role": "assistant",
            "llm.input_messages.1.message.content": "previous answer",
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call-1",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "move",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{}",
            "llm.tools.0.tool.json_schema": json.dumps(
                {"type": "function", "function": {"name": "move", "parameters": {}}}
            ),
            "llm.invocation_parameters": json.dumps(
                {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            ),
        }
    )
    record = registry.convert([{"SpanAttributes": attrs}])[0][0]
    assert record["tools"][0]["function"]["parameters"] == {}
    assert record["messages"][1]["weight"] == 0
    assert record["messages"][-1]["tool_calls"][0]["id"] == "call-1"
    assert record["chat_template_kwargs"] == {"enable_thinking": False}
    attrs.pop("llm.tools.0.tool.json_schema")
    with pytest.raises(ValueError, match="captured tool schemas"):
        registry.convert([{"SpanAttributes": attrs}])


@pytest.mark.parametrize(
    "attrs",
    [
        {"openinference.span.kind": "LLM", "input.value": "summary", "output.value": "answer"},
        {"openinference.span.kind": "LLM", "llm.token_count.total": 10},
        {
            **attributes(),
            "llm.output_messages.1.message.role": "assistant",
            "llm.output_messages.1.message.content": "other choice",
        },
    ],
)
def test_incomplete_or_ambiguous_spans_fail_with_context(attrs):
    with pytest.raises(ValueError, match=r"Span call \[openinference/v1\]"):
        registry.convert([{"SpanId": "call", "SpanAttributes": attrs}])


def test_nested_openinference_export_preserves_json_tool_schemas():
    rows = [
        {
            "context": {"trace_id": "trace", "span_id": "call"},
            "attributes": {
                "openinference": {"span": {"kind": "LLM"}},
                "llm": {
                    "input_messages": [{"message": {"role": "user", "content": "go"}}],
                    "output_messages": [{"message": {"role": "assistant", "content": "north"}}],
                    "tools": [
                        {
                            "tool": {
                                "json_schema": {
                                    "type": "function",
                                    "function": {"name": "move", "parameters": {}},
                                }
                            }
                        }
                    ],
                },
            },
        }
    ]
    (trace,) = traces_from_spans(rows)
    (record,) = TraceSpec().iter_chats(trace)
    assert record["tools"][0]["function"]["name"] == "move"


def test_provider_exports_must_retain_openinference_attributes():
    with pytest.raises(ValueError, match="OpenInference"):
        list(traces_from_spans([{"trace_id": "trace", "id": "call", "inputs": "summary"}]))


def test_dataframe_missing_attributes_do_not_create_phantom_tool_calls():
    first = {
        "context.trace_id": "trace",
        "context.span_id": "first",
        **{"attributes." + key: value for key, value in attributes().items()},
        "attributes.llm.output_messages.0.message.tool_calls.0.tool_call.id": "call",
        "attributes.llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "move",
        "attributes.llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{}",
        "attributes.llm.tools.0.tool.json_schema": json.dumps(
            {"type": "function", "function": {"name": "move", "parameters": {}}}
        ),
    }
    second = {
        "context.trace_id": "trace",
        "context.span_id": "second",
        **{"attributes." + key: value for key, value in attributes().items()},
    }
    (trace,) = traces_from_spans(frame_records(pd.DataFrame([first, second])))
    records = list(TraceSpec().iter_chats(trace))
    assert len(records) == 2
    assert records[1]["tools"] is None
    assert "tool_calls" not in records[1]["messages"][-1]


def test_nested_content_and_tool_call_arrays_are_flattened():
    rows = [
        {
            "trace_id": "trace",
            "span_id": "call",
            "attributes": {
                "openinference.span.kind": "LLM",
                "llm.input_messages": [
                    {
                        "message": {
                            "role": "user",
                            "contents": [{"message_content": {"type": "text", "text": "go"}}],
                        }
                    }
                ],
                "llm.output_messages": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "tool_call": {
                                        "id": "call",
                                        "function.name": "move",
                                        "function.arguments": "{}",
                                    }
                                }
                            ],
                        }
                    }
                ],
                "llm.tools": [
                    {
                        "tool": {
                            "json_schema": {
                                "type": "function",
                                "function": {"name": "move", "parameters": {}},
                            }
                        }
                    }
                ],
            },
        }
    ]
    (trace,) = traces_from_spans(rows)
    (record,) = TraceSpec().iter_chats(trace)
    assert record["messages"][0]["content"] == "go"
    assert record["messages"][-1]["tool_calls"][0]["function"]["name"] == "move"


@pytest.mark.parametrize("content", ["123", "true", "null", "{}", "[1, 2]"])
def test_openinference_message_text_is_never_json_decoded(content):
    attrs = {
        **attributes(),
        "llm.input_messages.0.message.content": content,
        "llm.output_messages.0.message.content": content,
    }
    (trace,) = traces_from_spans([{"trace_id": "trace", "span_id": "call", "attributes": attrs}])
    (record,) = TraceSpec().iter_chats(trace)
    assert record["messages"][0]["content"] == content
    assert record["messages"][-1]["content"] == content


def test_sampling_provenance_comes_from_each_model_call():
    attrs = {
        **attributes(),
        "llm.invocation_parameters": json.dumps(
            {"temperature": 1, "top_p": 1, "extra_body": {"top_k": -1, "min_p": 0}}
        ),
    }
    trace = TraceRecord(
        id="trace",
        spans=[{"SpanId": "call", "SpanAttributes": attrs}],
        metadata={"experiment_id": "experiment"},
    )
    (record,) = TraceSpec().iter_chats(trace)
    metadata = json.loads(record["metadata_json"])
    assert metadata["sampling"] == {"temperature": 1, "top_p": 1, "top_k": -1, "min_p": 0}
    assert metadata["experiment_id"] == "experiment"


def test_responses_text_and_function_items_form_one_assistant_turn():
    attrs = {
        **attributes(),
        "output.value": {"object": "response", "status": "completed"},
        "llm.input_messages.1.message.role": "assistant",
        "llm.input_messages.1.message.content": "Looking it up.",
        "llm.input_messages.2.message.role": "assistant",
        "llm.input_messages.2.message.tool_calls.0.tool_call.id": "old-call",
        "llm.input_messages.2.message.tool_calls.0.tool_call.function.name": "move",
        "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments": "{}",
        "llm.input_messages.3.message.role": "tool",
        "llm.input_messages.3.message.tool_call_id": "old-call",
        "llm.input_messages.3.message.content": "north",
        "llm.output_messages.1.message.role": "assistant",
        "llm.output_messages.1.message.tool_calls.0.tool_call.id": "new-call",
        "llm.output_messages.1.message.tool_calls.0.tool_call.function.name": "move",
        "llm.output_messages.1.message.tool_calls.0.tool_call.function.arguments": "{}",
        "llm.tools.0.tool.json_schema": {
            "type": "function",
            "function": {"name": "move", "parameters": {}},
        },
    }
    record = registry.convert([{"SpanId": "response", "SpanAttributes": attrs}])[0][0]
    assert [message["role"] for message in record["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert record["messages"][1]["content"] == "Looking it up."
    assert record["messages"][1]["tool_calls"][0]["id"] == "old-call"
    assert record["messages"][1]["weight"] == 0
    assert record["messages"][-1]["content"] == "north"
    assert record["messages"][-1]["tool_calls"][0]["id"] == "new-call"


def test_standard_tool_schema_preserves_responses_function_definitions():
    attrs = {
        **attributes(),
        "llm.tools.0.tool.json_schema": {
            "type": "function",
            "name": "move",
            "description": "Move north",
            "parameters": {"type": "object"},
            "strict": True,
        },
    }
    record = registry.convert([{"SpanAttributes": attrs}])[0][0]
    assert record["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "move",
                "description": "Move north",
                "parameters": {"type": "object"},
                "strict": True,
            },
        }
    ]


@pytest.mark.parametrize("flags", [{}, {"strict": False, "defer_loading": True}])
def test_standard_tool_schema_preserves_anthropic_function_definitions(flags):
    attrs = {
        **attributes(),
        "llm.tools.0.tool.json_schema": {
            "name": "move",
            "description": "Move north",
            "input_schema": {"type": "object", "properties": {}},
            **flags,
        },
    }
    record = registry.convert([{"SpanAttributes": attrs}])[0][0]
    assert record["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "move",
                "description": "Move north",
                "parameters": {"type": "object", "properties": {}},
                **flags,
            },
        }
    ]


def test_anthropic_tool_parameter_renaming_preserves_template_visible_key_order():
    schema = {
        "name": "lookup",
        "input_schema": {"type": "object", "properties": {}},
        "description": "Find a deferred tool",
        "defer_loading": True,
    }
    attrs = {**attributes(), "llm.tools.0.tool.json_schema": json.dumps(schema)}
    record = registry.convert([{"SpanAttributes": attrs}])[0][0]
    assert list(record["tools"][0]["function"]) == [
        "name",
        "parameters",
        "description",
        "defer_loading",
    ]
    assert record["tools"][0]["function"]["defer_loading"] is True


def test_original_anthropic_request_and_response_values_are_replayable():
    attrs = {
        "openinference.span.kind": "LLM",
        "llm.system": "anthropic",
        "input.value": {
            "system": [{"type": "text", "text": "Be helpful."}],
            "tools": [{"name": "move", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "Go north"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Moving."},
                        {"type": "tool_use", "id": "call", "name": "move", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call", "content": "Moved"},
                        {"type": "text", "text": "Thanks"},
                    ],
                },
            ],
        },
        "output.value": {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
    }
    record = registry.convert([{"SpanId": "call", "SpanAttributes": attrs}])[0][0]
    assert [m["role"] for m in record["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
    assert record["messages"][0]["content"] == "Be helpful."
    assert record["messages"][2]["tool_calls"][0]["id"] == "call"
    assert record["messages"][2]["weight"] == 0
    assert record["messages"][-1]["content"] == "Done."
    assert record["tools"][0]["function"]["name"] == "move"


def test_responses_empty_arguments_omitted_by_provider_recover_by_call_id():
    attrs = attributes()
    attrs.update(
        {
            "llm.input_messages.1.message.role": "assistant",
            "llm.input_messages.1.message.tool_calls.0.tool_call.id": "call-empty",
            "llm.input_messages.1.message.tool_calls.0.tool_call.function.name": "poll",
            "llm.input_messages.2.message.role": "tool",
            "llm.input_messages.2.message.tool_call_id": "call-empty",
            "llm.input_messages.2.message.content": "unsupported call: poll",
            "input.value": {
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call-other",
                        "name": "poll",
                        "arguments": '{"id":1}',
                    },
                    {
                        "type": "function_call",
                        "call_id": "call-empty",
                        "name": "poll",
                        "arguments": "{}",
                    },
                ]
            },
            "llm.tools.0.tool.json_schema": {
                "type": "function",
                "function": {"name": "poll", "parameters": {}},
            },
        }
    )
    row = registry.convert([{"SpanAttributes": attrs}])[0][0]
    assert row["messages"][1]["tool_calls"][0]["function"]["arguments"] == {}
    assert row["messages"][1]["weight"] == 0
    attrs["input.value"]["input"].pop()
    with pytest.raises(ValueError, match="without a matching captured call"):
        registry.convert([{"SpanAttributes": attrs}])
