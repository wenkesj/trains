import json
import math
from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict, cast

type MessageRole = Literal["system", "developer", "user", "assistant", "tool"]
type JSONSchema = dict[str, object]


class FunctionCall(TypedDict):
    """A function name and its structured or serialized arguments."""

    name: str
    arguments: str | dict[str, object]


class ToolCall(TypedDict):
    """An assistant tool invocation with its call identity."""

    function: FunctionCall
    id: NotRequired[str]
    type: NotRequired[Literal["function"]]


class Message(TypedDict):
    """An HF chat message with content, tool calls, and optional reasoning."""

    role: MessageRole
    content: NotRequired[str | list[dict[str, Any]] | None]
    reasoning_content: NotRequired[str | None]
    reasoning: NotRequired[str | None]
    weight: NotRequired[float]
    name: NotRequired[str]
    tool_calls: NotRequired[list[ToolCall] | None]
    tool_call_id: NotRequired[str]


class FunctionDefinition(TypedDict):
    """A callable tool schema supplied to the model."""

    name: str
    description: NotRequired[str]
    parameters: NotRequired[JSONSchema]
    strict: NotRequired[bool | None]
    defer_loading: NotRequired[bool]


class ToolDefinition(TypedDict):
    """A typed tool wrapper containing a function definition."""

    type: Literal["function"]
    function: FunctionDefinition


class RenderingOptions(TypedDict, total=False):
    tools: list[ToolDefinition] | None
    chat_template_kwargs: dict[str, object] | None


class ConversationalRecord(RenderingOptions):
    """A full conversation with optional tools and chat-template settings."""

    messages: list[Message]


type HFRecord = ConversationalRecord


def validate_record(record: HFRecord | dict[str, Any]) -> HFRecord:
    """Accept HF conversations; tokenization controls assistant-only supervision."""
    raw = cast(dict[str, Any], record)
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Provide a nonempty messages conversation")
    checked = [validate_message(message) for message in messages]
    if not any(m["role"] == "assistant" and m.get("weight", 1) > 0 for m in checked):
        raise ValueError("Conversation requires a supervised assistant message")
    _validate_tools(checked)
    for tool in raw.get("tools") or []:
        validate_tool_definition(tool)
    if {
        "tokenize",
        "return_dict",
        "add_generation_prompt",
        "continue_final_message",
        "tools",
        "return_assistant_tokens_mask",
        "return_tensors",
        "padding",
        "truncation",
        "max_length",
    }.intersection(raw.get("chat_template_kwargs") or {}):
        raise ValueError("Rendering options cannot override tokenization or message boundaries")
    return {**raw, "messages": checked}


def model_call(messages: list[Message], output: Message) -> list[Message]:
    """Keep captured context while supervising only this call's generated assistant turn."""
    return [
        {**message, "weight": 0.0}
        if message["role"] == "assistant"
        else cast(Message, dict(message))
        for message in messages
    ] + [output]


def normalize_message(value: Mapping[str, Any]) -> Message:
    content = value.get("content")
    if isinstance(content, list):
        if not all(c.get("type") in {"text", "input_text", "output_text"} for c in content):
            return validate_message(value)
        return validate_message({**value, "content": "".join(c["text"] for c in content)})
    return validate_message(value)


def validate_message(value: Mapping[str, Any]) -> Message:
    """Validate untrusted native message fields and retain unknown provider metadata."""
    if not isinstance(value, Mapping):
        raise ValueError("Messages must be objects")
    if value.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
        raise ValueError("Unsupported chat role")
    if value.get("content") is not None and not isinstance(value["content"], (str, list)):
        raise ValueError("Message content must be text or ordered content parts")
    if isinstance(value.get("content"), list):
        for part in value["content"]:
            if not isinstance(part, dict) or not isinstance(part.get("type"), str):
                raise ValueError("Content parts require a type")
    for key in ("name", "tool_call_id"):
        if key in value and not isinstance(value[key], str):
            raise ValueError(f"Message {key} must be a string")
    for key in ("reasoning", "reasoning_content"):
        if value.get(key) is not None and (
            value["role"] != "assistant" or not isinstance(value[key], str)
        ):
            raise ValueError("Reasoning must be assistant text")
    if (
        value.get("reasoning")
        and value.get("reasoning_content")
        and value["reasoning"] != value["reasoning_content"]
    ):
        raise ValueError("Conflicting reasoning and reasoning_content")
    weight = value.get("weight", 1)
    if type(weight) not in {int, float} or not math.isfinite(weight) or weight < 0:
        raise ValueError("Message weight must be finite and nonnegative")
    if "weight" in value and value["role"] != "assistant":
        raise ValueError("Only assistant messages carry supervision weights")
    _validate_calls(value)
    if value["role"] == "assistant" and not any(
        value.get(k) for k in ("content", "tool_calls", "reasoning", "reasoning_content")
    ):
        raise ValueError("Assistant output has no content, reasoning, or tool calls")
    return cast(Message, dict(value))


def validate_tool_definition(value: Mapping[str, Any]) -> ToolDefinition:
    function = value.get("function")
    if (
        value.get("type") != "function"
        or not isinstance(function, dict)
        or not isinstance(function.get("name"), str)
    ):
        raise ValueError("Tool definitions require a named function")
    if "parameters" in function and not isinstance(function["parameters"], dict):
        raise ValueError("Tool parameters must be a JSON schema object")
    if "description" in function and not isinstance(function["description"], str):
        raise ValueError("Tool description must be a string")
    if function.get("strict") is not None and not isinstance(function["strict"], bool):
        raise ValueError("Tool strict must be a boolean or null")
    if "defer_loading" in function and not isinstance(function["defer_loading"], bool):
        raise ValueError("Tool defer_loading must be a boolean")
    return cast(ToolDefinition, dict(value))


def _validate_tools(messages: list[Message]) -> None:
    pending: set[str] = set()
    seen: set[str] = set()
    for index, message in enumerate(messages):
        if message["role"] == "tool":
            call = message.get("tool_call_id")
            if call is None or call not in pending:
                raise ValueError("Orphan or duplicate tool result")
            pending.remove(call)
            continue
        if pending:
            raise ValueError("Missing tool results before the next model turn")
        calls = message.get("tool_calls") or []
        for call in calls:
            identifier = call.get("id")
            if identifier is None:
                if index != len(messages) - 1:
                    raise ValueError("Context tool calls require IDs for matching tool results")
            elif not identifier or identifier in seen:
                raise ValueError("Duplicate or empty tool call ID")
            else:
                seen.add(identifier)
                pending.add(identifier)
    if pending and messages[-1]["role"] != "assistant":
        raise ValueError("Conversation ends before pending tool results")


def _validate_calls(value: Mapping[str, Any]) -> None:
    calls = value.get("tool_calls")
    if calls is not None and not isinstance(calls, list):
        raise ValueError("tool_calls must be a list or null")
    for call in calls or []:
        if value["role"] != "assistant" or not isinstance(call, dict):
            raise ValueError("Tool calls require an assistant message and function objects")
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("Tool calls require a named function")
        if not isinstance(function.get("arguments"), (str, dict)):
            raise ValueError("Tool arguments must be JSON text or an object")
        arguments = function["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError as exc:
                raise ValueError("Tool arguments must be valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        if call.get("type", "function") != "function" or (
            "id" in call and not isinstance(call["id"], str)
        ):
            raise ValueError("Invalid tool call type or ID")


__all__ = [
    "ConversationalRecord",
    "FunctionCall",
    "FunctionDefinition",
    "HFRecord",
    "JSONSchema",
    "Message",
    "MessageRole",
    "ToolCall",
    "ToolDefinition",
    "validate_record",
]
