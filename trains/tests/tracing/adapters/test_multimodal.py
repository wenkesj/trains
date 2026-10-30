from trains.tracing.adapters.registry import registry


def test_openinference_flattened_parts_retain_order_and_media():
    attrs = {
        "openinference.span.kind": "LLM",
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.contents.0.message_content.type": "text",
        "llm.input_messages.0.message.contents.0.message_content.text": "describe",
        "llm.input_messages.0.message.contents.1.message_content.type": "image",
        "llm.input_messages.0.message.contents.1.message_content.image.image.url": "data:image/png;base64,cG5n",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "a picture",
    }
    values, _ = registry.convert([{"SpanId": "call", "SpanAttributes": attrs}])
    parts = values[0]["messages"][:-1][0]["content"]
    assert [part["type"] for part in parts] == ["text", "image"]
    assert parts[1]["image"].startswith("data:image/png")
