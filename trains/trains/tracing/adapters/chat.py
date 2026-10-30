from collections.abc import Mapping
from typing import Any, cast

from trains.formats.records import (
    HFRecord,
    Message,
    ToolCall,
    ToolDefinition,
    model_call,
    normalize_message,
    validate_message,
    validate_record,
    validate_tool_definition,
)
from trains.tracing.adapters.span import Span, decoded


class ChatAdapter:
    """Decode OpenInference chat, Responses, and Anthropic model-call spans."""

    id = "openinference/v1"
    description = "OpenInference model calls"

    @staticmethod
    def matches(span: Span) -> bool:
        """Return whether the span contains the model-call format handled here."""
        return span.attributes.get("openinference.span.kind") == "LLM"

    @staticmethod
    def extract(span: Span) -> list[HFRecord]:
        """Convert the matched span into complete HF model-call records."""
        attrs = span.attributes
        inputs = attrs.get("input.value")
        raw_output = attrs.get("output.value")
        anthropic = "anthropic" in (attrs.get("llm.system"), attrs.get("llm.provider"))
        responses = isinstance(raw_output, dict) and raw_output.get("object") == "response"
        messages = _flattened(attrs, "llm.input_messages.", inputs)
        output_messages = _flattened(attrs, "llm.output_messages.", raw_output)
        if responses:
            messages = _assistant_turns(messages)
            output_messages = _assistant_turns(output_messages)
        if len(output_messages) > 1:
            raise ValueError("Multiple output choices require explicit selection")
        outputs = (
            output_messages[0] if output_messages else _single_output(attrs.get("output.value"))
        )
        if not messages:
            messages = inputs.get("messages") if isinstance(inputs, dict) else inputs
        options = attrs.get("llm.invocation_parameters") or {}
        tools = (
            _span_tools(attrs)
            or (inputs.get("tools") if isinstance(inputs, dict) else None)
            or options.get("tools")
            or None
        )
        tools = [_tool_schema(tool) for tool in tools] if tools else None
        if anthropic and not _flattened(attrs, "llm.input_messages.", inputs):
            messages = _anthropic_messages(inputs)
        if anthropic and not output_messages:
            outputs = _anthropic_message(outputs)
        messages = (
            [normalize_message(m) for m in messages] if isinstance(messages, list) else messages
        )
        outputs = normalize_message(outputs) if isinstance(outputs, dict) else outputs
        if not isinstance(messages, list) or not messages or not isinstance(outputs, dict):
            raise ValueError(
                "OpenInference model call needs captured messages and a single assistant output"
            )
        if outputs.get("role") != "assistant":
            raise ValueError("OpenInference model output must have the assistant role")
        if any(m.get("tool_calls") for m in [*messages, outputs]) and not tools:
            raise ValueError("Tool-calling model spans need captured tool schemas")
        return [
            validate_record(
                {
                    "tools": tools,
                    "chat_template_kwargs": _rendering_options(inputs, options),
                    "messages": model_call(messages, outputs),
                }
            )
        ]


def _rendering_options(inputs: Any, options: Mapping[str, Any]) -> dict[str, Any]:
    """Read chat-template settings from model invocation parameters."""
    return (
        (
            inputs.get("chat_template_kwargs")
            or (inputs.get("extra_body") or {}).get("chat_template_kwargs", {})
            if isinstance(inputs, dict)
            else {}
        )
        or options.get("extra_body", {}).get("chat_template_kwargs", {})
        or options.get("chat_template_kwargs", {})
    )


def _flattened(attrs: dict, prefix: str, payload: Any = None) -> list[Message]:
    """Rebuild indexed OpenInference message attributes in their original order."""
    indices = sorted({int(k[len(prefix) :].split(".")[0]) for k in attrs if k.startswith(prefix)})
    result: list[Message] = []
    for index in indices:
        root = f"{prefix}{index}.message."
        value: dict[str, Any] = {
            "role": attrs.get(root + "role"),
            "content": attrs.get(root + "content"),
        }
        parts = _flattened_parts(attrs, root)
        if parts:
            value["content"] = parts
        calls: list[ToolCall] = []
        numbers = sorted(
            {
                int(k[len(root + "tool_calls.") :].split(".")[0])
                for k in attrs
                if k.startswith(root + "tool_calls.")
            }
        )
        for number in numbers:
            call = root + f"tool_calls.{number}.tool_call."
            calls.append(
                {
                    **({"id": attrs[call + "id"]} if call + "id" in attrs else {}),
                    "type": "function",
                    "function": {
                        "name": attrs[call + "function.name"],
                        "arguments": _function_arguments(attrs, call, payload),
                    },
                }
            )
        if calls:
            value["tool_calls"] = calls
        if root + "tool_call_id" in attrs:
            value["tool_call_id"] = attrs[root + "tool_call_id"]
        result.append(validate_message(value))
    return result


def _function_arguments(attrs: dict, call: str, payload: Any) -> Any:
    """Recover omitted Responses arguments only from the captured matching call."""
    key = call + "function.arguments"
    if key in attrs:
        return decoded(attrs[key])
    identifier = attrs.get(call + "id")
    if isinstance(identifier, str) and identifier and isinstance(payload, Mapping):
        for field in ("input", "output"):
            items = payload.get(field)
            if not isinstance(items, list):
                continue
            for item in items:
                if (
                    isinstance(item, Mapping)
                    and item.get("type") == "function_call"
                    and item.get("call_id") == identifier
                    and item.get("name") == attrs.get(call + "function.name")
                    and "arguments" in item
                ):
                    return decoded(item["arguments"])
    raise ValueError("Traced function arguments are missing without a matching captured call")


def _flattened_parts(attrs: Mapping[str, Any], root: str) -> list[dict[str, Any]]:
    """Reconstruct indexed message content parts from span attributes."""
    parts = []
    content_root = root + "contents."
    content_indices = sorted(
        {
            int(key[len(content_root) :].split(".")[0])
            for key in attrs
            if key.startswith(content_root)
        }
    )
    for content_index in content_indices:
        part_root = content_root + f"{content_index}.message_content."
        kind = attrs.get(part_root + "type")
        if kind == "text":
            parts.append({"type": "text", "text": attrs.get(part_root + "text", "")})
        elif kind in {"image", "audio", "video"}:
            parts.append({"type": kind, kind: attrs.get(part_root + kind + "." + kind + ".url")})
        elif kind == "tool_use":
            # Anthropic's instrumentor repeats each tool call in the ordered contents
            # and the standard message.tool_calls array. Retain the latter once.
            call = content_root + f"{content_index}.tool_call."
            fields = ("id", "function.name", "function.arguments")
            numbers = {
                key.removeprefix(root + "tool_calls.").split(".")[0]
                for key in attrs
                if key.startswith(root + "tool_calls.")
            }
            if not any(
                all(
                    attrs.get(call + field)
                    == attrs.get(root + f"tool_calls.{number}.tool_call." + field)
                    for field in fields
                )
                for number in numbers
            ):
                raise ValueError("Traced tool_use content has no matching standard tool call")
        else:
            raise ValueError(f"Unsupported traced content part: {kind}")
    return parts


def _single_output(outputs: Any) -> Any:
    """Require exactly one complete model output for replay."""
    if isinstance(outputs, dict) and "choices" in outputs:
        if len(outputs["choices"]) != 1:
            raise ValueError("Multiple output choices require explicit selection")
        outputs = outputs["choices"][0].get("message")
    if isinstance(outputs, list):
        if len(outputs) != 1:
            raise ValueError("Multiple output choices require explicit selection")
        outputs = outputs[0]
    if isinstance(outputs, str):
        outputs = {"role": "assistant", "content": outputs}
    return outputs


def _span_tools(attrs: dict) -> list[ToolDefinition]:
    """Read the tool definitions visible to this model call."""
    indices = sorted({int(k.split(".")[2]) for k in attrs if k.startswith("llm.tools.")})
    result = []
    for index in indices:
        schema = decoded(attrs[f"llm.tools.{index}.tool.json_schema"])
        result.append(_tool_schema(schema))
    return result


def _tool_schema(schema: Any) -> ToolDefinition:
    """Normalize native tool fields while preserving template-visible schema order."""
    if isinstance(schema, dict) and "name" in schema and "input_schema" in schema:
        schema = {
            "type": "function",
            "function": {
                "parameters" if key == "input_schema" else key: value
                for key, value in schema.items()
                if key in {"name", "description", "input_schema", "strict", "defer_loading"}
            },
        }
    if isinstance(schema, dict) and schema.get("type") == "function" and "name" in schema:
        schema = {
            "type": "function",
            "function": {key: value for key, value in schema.items() if key != "type"},
        }
    return validate_tool_definition(schema)


def _assistant_turns(messages: list[Message]) -> list[Message]:
    """Join contiguous assistant items belonging to one Responses API turn."""
    result: list[Message] = []
    for message in messages:
        if not result or message["role"] != "assistant" or result[-1]["role"] != "assistant":
            result.append(cast(Message, dict(message)))
            continue
        previous = result[-1]
        content, extra = previous.get("content") or "", message.get("content") or ""
        if isinstance(content, str) and isinstance(extra, str):
            previous["content"] = content + extra
        else:
            previous["content"] = (
                content if isinstance(content, list) else [{"type": "text", "text": content}]
            ) + (extra if isinstance(extra, list) else [{"type": "text", "text": extra}])
        calls = [*(previous.get("tool_calls") or []), *(message.get("tool_calls") or [])]
        if calls:
            previous["tool_calls"] = calls
        reasoning = (previous.get("reasoning_content") or "") + (
            message.get("reasoning_content") or ""
        )
        if reasoning:
            previous["reasoning_content"] = reasoning
    return result


def _anthropic_message(message: Mapping[str, Any]) -> Message:
    """Normalize the captured native Anthropic content blocks without generating text."""
    if not isinstance(message, Mapping):
        raise ValueError("Anthropic model calls require a captured output object")
    if isinstance(message.get("content"), str):
        return validate_message(message)
    parts, calls, thinking = [], [], []
    for block in message.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": block["text"]})
        elif kind == "tool_use":
            calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": block["input"],
                    },
                }
            )
        elif kind == "thinking":
            thinking.append(block["thinking"])
        else:
            raise ValueError(f"Unsupported captured Anthropic content block: {kind}")
    return validate_message(
        {
            "role": message["role"],
            "content": parts or None,
            **({"tool_calls": calls} if calls else {}),
            **({"reasoning_content": "".join(thinking)} if thinking else {}),
        }
    )


def _anthropic_messages(request: Any) -> list[Message]:
    """Convert Anthropic content blocks into complete HF chat messages."""
    if not isinstance(request, dict):
        raise ValueError("Anthropic model calls require the captured request object")
    result = []
    if request.get("system"):
        result.append(_anthropic_message({"role": "system", "content": request["system"]}))
    for message in request["messages"]:
        blocks = message.get("content")
        if message["role"] != "user" or not isinstance(blocks, list):
            result.append(_anthropic_message(message))
            continue
        pending = []
        for block in blocks:
            if block.get("type") == "tool_result":
                if pending:
                    result.append(_anthropic_message({"role": "user", "content": pending}))
                    pending = []
                result.append(
                    validate_message(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block.get("content"),
                        }
                    )
                )
            else:
                pending.append(block)
        if pending:
            result.append(_anthropic_message({"role": "user", "content": pending}))
    return result
