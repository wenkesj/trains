from types import SimpleNamespace

import torch
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.renderers.qwen3 import Qwen3VLRenderer

from trains.datum.adapters.cookbook import CookbookRenderer, _cookbook_message


def test_cookbook_wrapper_uses_all_assistant_policy():
    called = []
    renderer = SimpleNamespace(
        build_supervised_examples=lambda messages, policy: called.append(policy) or []
    )
    assert list(CookbookRenderer(renderer).build_supervised_examples([])) == []
    assert called == [TrainOnWhat.ALL_ASSISTANT_MESSAGES]


def test_cookbook_preserves_reasoning_tools_and_selected_turns():
    called = []
    renderer = SimpleNamespace(
        build_supervised_examples=lambda messages, policy: (
            called.append((messages, policy)) or [("input", torch.tensor([0.0, 1.0]))]
        )
    )
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "old", "weight": 0},
        {"role": "user", "content": "again"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "plan",
            "weight": 0.5,
            "tool_calls": [{"id": "a", "function": {"name": "lookup", "arguments": {"x": 1}}}],
        },
    ]
    examples = CookbookRenderer(renderer).build_supervised_examples(messages)
    assert len(examples) == 1 and examples[0][1].tolist() == [0.0, 0.5]
    native, policy = called[0]
    assert policy == TrainOnWhat.LAST_ASSISTANT_MESSAGE
    assert native[-1]["content"][0] == {"type": "thinking", "thinking": "plan"}
    assert native[-1]["tool_calls"][0].function.arguments == '{"x": 1}'
    assert "reasoning_content" not in native[-1]


def test_cookbook_tool_schemas_use_native_prefix_and_keep_system():
    seen = []
    renderer = SimpleNamespace(
        create_conversation_prefix_with_tools=lambda schemas, system: (
            seen.append((schemas, system)) or [{"role": "system", "content": system + " tools"}]
        ),
        build_supervised_examples=lambda messages, policy: seen.append(messages) or [],
    )
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "answer"},
    ]
    schemas = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}, "strict": True},
        }
    ]
    assert CookbookRenderer(renderer).build_supervised_examples(messages, tools=schemas) == []
    assert seen[0][0][0]["name"] == "lookup" and seen[0][0][0]["strict"] is True
    assert seen[0][1] == "instructions"
    assert seen[1][0] == {"role": "system", "content": "instructions tools"}
    assert messages[0]["content"] == "instructions"


def test_openai_image_parts_reach_native_vision_renderer():
    image = "data:image/png;base64,cG5n"
    message = _cookbook_message(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Read the chart"},
                {"type": "image_url", "image_url": {"url": image}},
            ],
        }
    )
    assert message["content"][1] == {"type": "image", "image": image}
    # Exercise native vision content handling: unconverted image_url parts would be omitted.
    renderer = object.__new__(Qwen3VLRenderer)
    parts = renderer._preprocess_message_parts(message)
    assert any(part == {"type": "image", "image": image} for part in parts)
