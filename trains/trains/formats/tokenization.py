import copy
import json
import re
from typing import Any

from trains.formats.records import HFRecord, Message, validate_record
from trains.formats.tokens import conversation_datum, encoded_datum


def encode(record: HFRecord, tokenizer: Any) -> tuple[list[int], list[float]]:
    """Render a conversation once for training, with loss on assistant spans only."""
    encoder: Any = getattr(tokenizer, "tokenizer", tokenizer)
    if encoder is None:
        raise ValueError("Supply the inference tokenizer as processing_class")
    record = validate_record(record)
    messages = _render_messages(record["messages"])
    kwargs = dict(record.get("chat_template_kwargs") or {})
    if record.get("tools"):
        kwargs["tools"] = record["tools"]
    _check_payloads(messages, encoder, kwargs)
    template = (
        encoder.get_chat_template()
        if hasattr(encoder, "get_chat_template")
        else getattr(encoder, "chat_template", "")
    )
    native_mask = isinstance(template, str) and re.search(r"\{%-?\s*generation\b", template)
    if native_mask and all(m.get("weight", 1) == 1 for m in messages):
        result = encoder.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            **kwargs,
        )
        tokens = list(result["input_ids"])
        generated = _assistant_weights(messages, tokens, encoder, kwargs)
        weights = [
            float(native) * action
            for native, action in zip(result["assistant_masks"], generated, strict=True)
        ]
    else:
        tokens = list(
            encoder.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_dict=False, **kwargs
            )
        )
        weights = _assistant_weights(messages, tokens, encoder, kwargs)
    if len(weights) != len(tokens) or not any(weights[1:]):
        raise ValueError("Chat template produced no aligned assistant supervision")
    return tokens, weights


def _render_messages(messages: list[Message]) -> list[Message]:
    """Convert wire JSON tool arguments to the mapping expected by HF templates."""
    result = copy.deepcopy(messages)
    for message in result:
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            message["reasoning_content"] = message["reasoning"] = reasoning
        if message.get("tool_calls") is None:
            message.pop("tool_calls", None)
        for call in message.get("tool_calls") or []:
            function = call["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                function["arguments"] = json.loads(arguments)
            if not isinstance(function.get("arguments"), dict):
                raise ValueError("Tool arguments must be a JSON object")
    return result


def _assistant_weights(
    messages: list[Message], tokens: list[int], encoder: Any, kwargs: dict
) -> list[float]:
    """Require exact prefix alignment when the template has no native generation mask."""
    weights = [0.0] * len(tokens)
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or not message.get("weight", 1):
            continue
        if index == 0:
            raise ValueError("Assistant supervision requires preceding context")
        prompt = encoder.apply_chat_template(
            messages[:index], tokenize=True, add_generation_prompt=True, return_dict=False, **kwargs
        )
        end = encoder.apply_chat_template(
            messages[: index + 1],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
            **kwargs,
        )
        if end[: len(prompt)] != prompt or tokens[: len(end)] != end:
            raise ValueError(
                "Chat template does not preserve the generation prefix; use a training template with generation masks"
            )
        if len(end) <= len(prompt):
            raise ValueError("Assistant message rendered no target tokens")
        stop = len(end)
        eos = getattr(encoder, "eos_token_id", None)
        if eos is not None:
            terminal = [
                position for position in range(len(prompt), len(end)) if end[position] == eos
            ]
            if terminal:
                stop = terminal[-1] + 1
        weights[len(prompt) : stop] = [float(message.get("weight", 1))] * (stop - len(prompt))
    return weights


def _check_payloads(messages: list[Message], encoder: Any, kwargs: dict) -> None:
    """Reject templates that silently discard reasoning, calls, or tool definitions."""
    payloads = [
        (index, label, fields)
        for index, message in enumerate(messages)
        for label, fields in (
            ("reasoning", ("reasoning_content", "reasoning")),
            ("tool calls", ("tool_calls",)),
        )
        if any(message.get(key) for key in fields)
    ]
    if not payloads and not kwargs.get("tools"):
        return
    rendered = encoder.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **kwargs
    )
    for index, label, fields in payloads:
        stripped = [dict(message) for message in messages]
        for field in fields:
            stripped[index].pop(field, None)
        without = encoder.apply_chat_template(
            stripped, tokenize=False, add_generation_prompt=False, **kwargs
        )
        if rendered == without:
            raise ValueError(
                f"Chat template drops {label}; supply a training template that renders every assistant turn"
            )
    if kwargs.get("tools"):
        without_tools = {key: value for key, value in kwargs.items() if key != "tools"}
        if rendered == encoder.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **without_tools
        ):
            raise ValueError("Chat template drops tool schemas; supply a tool-aware template")


__all__ = ["conversation_datum", "encode", "encoded_datum"]
