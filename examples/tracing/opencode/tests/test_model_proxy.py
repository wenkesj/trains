import asyncio
import json
from types import SimpleNamespace

from opencode_tracing.model_proxy import _invoke


def test_proxy_orders_schemas_before_sdk_instrumentation_and_preserves_messages():
    captured = {}

    async def create(*, model, messages, tools, extra_body, extra_headers):
        captured.update(
            model=model,
            messages=messages,
            tools=tools,
            extra_body=extra_body,
            extra_headers=extra_headers,
        )
        return "native-response"

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute",
                "description": "Run",
                "parameters": {
                    "type": "object",
                    "properties": {"z": {"type": "string"}, "a": {"type": "string"}},
                    "required": ["z", "a"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "Implement the exercise."}]
    body = {
        "model": "frozen-policy",
        "messages": messages,
        "tools": tools,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    original = json.dumps(body)
    result = asyncio.run(_invoke(client, "chat/completions", body, {"openai-beta": "test"}))
    assert result == "native-response"
    assert captured["messages"] is messages
    assert json.dumps(captured["tools"]) == json.dumps(
        json.loads(json.dumps(tools, sort_keys=True))
    )
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert captured["extra_headers"] == {"openai-beta": "test"}
    assert json.dumps(body) == original
