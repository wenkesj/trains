import pytest
from trains.tracing.adapters.chat import _flattened

from claude_tracing.model_proxy import _canonical_request


def test_anthropic_tool_use_repetition_is_preserved_once():
    root = "llm.input_messages.0.message."
    attrs = {
        root + "role": "assistant",
        root + "contents.0.message_content.type": "text",
        root + "contents.0.message_content.text": "Checking.",
        root + "contents.1.message_content.type": "tool_use",
    }
    fields = {"id": "call-1", "function.name": "execute", "function.arguments": {"command": "pwd"}}
    for field, value in fields.items():
        attrs[root + "contents.1.tool_call." + field] = value
        attrs[root + "tool_calls.0.tool_call." + field] = value
    messages = _flattened(attrs, "llm.input_messages.")
    assert messages[0]["content"] == [{"type": "text", "text": "Checking."}]
    calls = messages[0]["tool_calls"]
    assert calls is not None and len(calls) == 1
    attrs[root + "contents.1.tool_call.id"] = "other"
    with pytest.raises(ValueError, match="no matching"):
        _flattened(attrs, "llm.input_messages.")


def test_native_request_defaults_are_captured_before_backend_translation():
    request = {
        "system": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "system", "content": "inline"},
        ],
        "tools": [{"name": "execute", "input_schema": {"type": "object"}, "defer_loading": True}],
    }
    result = _canonical_request(request)
    assert result["system"] == "first\nsecond\ninline"
    assert result["messages"] == [{"role": "user", "content": "question"}]
    assert result["tools"][0]["strict"] is False
    assert result["tools"][0]["defer_loading"] is True
    assert list(result["tools"][0]) == [
        "name",
        "description",
        "input_schema",
        "strict",
        "defer_loading",
    ]
    assert isinstance(request["system"], list)
    assert "strict" not in request["tools"][0]
    assert _canonical_request(result) == result


def test_invalid_system_content_is_not_silently_discarded():
    with pytest.raises(ValueError, match="text blocks"):
        _canonical_request({"system": [{"type": "image", "source": {}}], "messages": []})
    assert "system" not in _canonical_request({"system": "  ", "messages": []})
