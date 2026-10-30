from trains.formats.records import HFRecord, model_call, normalize_message, validate_record


def chat_completion_record(request: dict, response: dict) -> HFRecord:
    """Convert one captured Chat Completions request/response without tracing metadata."""
    choices = response.get("choices") or []
    messages = request.get("messages")
    if len(choices) != 1 or not isinstance(messages, list) or not messages:
        raise ValueError("Captured call needs messages and exactly one response choice")
    return validate_record(
        {
            "tools": request.get("tools") or None,
            "chat_template_kwargs": request.get("chat_template_kwargs")
            or (request.get("extra_body") or {}).get("chat_template_kwargs", {}),
            "messages": model_call(
                [normalize_message(message) for message in messages],
                normalize_message(choices[0]["message"]),
            ),
        }
    )
