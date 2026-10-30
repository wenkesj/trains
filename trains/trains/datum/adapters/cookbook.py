import json
from collections.abc import Mapping
from typing import Any, cast

import torch
from tinker import types
from tinker_cookbook.renderers import Message, Renderer, ToolCall, ToolSpec, TrainOnWhat


class CookbookRenderer:
    """Render full conversations with the native model-specific assistant supervision."""

    supports_message_weights = True
    supports_rendering_options = True

    def __init__(self, renderer: Renderer) -> None:
        """Use a native Cookbook renderer to preserve its token and media layout.

        Args:
            renderer: Native renderer responsible for exact chat and media token layout.
        """
        self.renderer = renderer

    def build_supervised_examples(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[tuple[types.ModelInput, torch.Tensor]]:
        """Render complete messages while preserving the configured assistant masks."""
        if chat_template_kwargs:
            raise ValueError(
                "Configure thinking/rendering options on the native renderer; HF chat_template_kwargs cannot be silently ignored"
            )
        if tools:
            messages = _tool_prefix(messages, tools, self.renderer)
        native = [_cookbook_message(message) for message in messages]
        if all(message.get("weight", 1) == 1 for message in messages):
            return self.renderer.build_supervised_examples(
                native, TrainOnWhat.ALL_ASSISTANT_MESSAGES
            )
        result = []
        for index, message in enumerate(messages):
            weight = message.get("weight", 1) if message["role"] == "assistant" else 0
            if not weight:
                continue
            for model_input, weights in self.renderer.build_supervised_examples(
                native[: index + 1], TrainOnWhat.LAST_ASSISTANT_MESSAGE
            ):
                result.append((model_input, weights * weight))
        return result


def _cookbook_message(message: Mapping[str, Any]) -> Message:
    """Translate an HF message into the native Cookbook message structure."""
    value = {
        key: item
        for key, item in message.items()
        if key not in {"weight", "reasoning", "reasoning_content", "tool_calls"}
    }
    content = message.get("content") or ""
    if isinstance(content, list):
        content = [_cookbook_part(part) for part in content]
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        value["content"] = [{"type": "thinking", "thinking": reasoning}, *parts]
    else:
        value["content"] = content
    if message.get("tool_calls"):
        value["tool_calls"] = [
            ToolCall.model_validate(
                {
                    **call,
                    "function": {
                        **call["function"],
                        "arguments": json.dumps(call["function"]["arguments"])
                        if isinstance(call["function"]["arguments"], dict)
                        else call["function"]["arguments"],
                    },
                }
            )
            for call in message["tool_calls"]
        ]
    return cast(Message, value)


def _cookbook_part(part: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a content part without discarding native media."""
    kind = part.get("type")
    if kind in {"text", "input_text", "output_text"}:
        return {"type": "text", "text": part["text"]}
    if kind in {"image_url", "input_image", "image"}:
        image = part.get("image_url") or part.get("image") or part.get("url")
        if isinstance(image, Mapping):
            image = image["url"]
        if image is None:
            raise ValueError("Image content requires an image URI or native image object")
        return {"type": "image", "image": image}
    if kind in {"thinking", "audio"}:
        return dict(part)
    raise ValueError(f"Unsupported native cookbook content part: {kind}")


def _tool_prefix(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]], renderer: Renderer
) -> list[dict[str, Any]]:
    """Render the tool-definition prefix required by the native renderer."""
    system = ""
    if messages and messages[0]["role"] == "system":
        system = messages[0].get("content") or ""
        if not isinstance(system, str):
            raise ValueError("Native tool-prefix rendering requires a text system message")
        messages = messages[1:]
    schemas = [
        cast(
            ToolSpec,
            {
                **tool["function"],
                "description": tool["function"].get("description", ""),
                "parameters": tool["function"].get("parameters", {}),
            },
        )
        for tool in tools
    ]
    prefix = renderer.create_conversation_prefix_with_tools(schemas, system)
    return [dict(message) for message in prefix] + messages
