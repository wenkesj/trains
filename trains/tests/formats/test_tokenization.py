import json
import string

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

from trains.datasets.serialization import datum_mapping
from trains.datum.adapters.baseten import BasetenDatumAdapter
from trains.datum.adapters.fireworks import FireworksDatumAdapter
from trains.datum.adapters.spindle import SpindleDatumAdapter
from trains.datum.adapters.tinker import TinkerDatumAdapter
from trains.formats.records import validate_record
from trains.formats.tokenization import _render_messages, conversation_datum, encode


@pytest.fixture
def tokenizer():
    vocabulary = {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(string.printable)}}
    raw = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    raw.pre_tokenizer = Split("", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    tokenizer.chat_template = """{% if tools %}{{ tools | tojson }}{% endif %}{% for m in messages %}{{ m.role + ':' }}{% if m.role == 'assistant' %}{% generation %}{% if m.reasoning_content %}{{ '<think>' + m.reasoning_content + '</think>' }}{% endif %}{{ m.content or '' }}{% for call in m.tool_calls | default([]) %}{{ call.function.name + (call.function.arguments | tojson) }}{% endfor %}{{ '|' }}{% endgeneration %}{% else %}{{ m.content + '|' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant:' }}{% endif %}"""
    return tokenizer


def conversation():
    return {
        "messages": [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "plan",
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "tool-result"},
            {"role": "assistant", "reasoning": "check", "content": "answer"},
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "final"},
        ],
        "tools": [
            {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
        ],
    }


def supervised_text(tokenizer, tokens, weights):
    return "".join(
        tokenizer.convert_ids_to_tokens([t for t, w in zip(tokens, weights, strict=True) if w])
    )


def test_multi_turn_reasoning_and_tools_have_one_aligned_datum(tokenizer):
    record = conversation()
    original = json.loads(json.dumps(record))
    tokens, weights = encode(record, tokenizer)
    text = supervised_text(tokenizer, tokens, weights)
    assert text == '<think>plan</think>lookup{"city": "Paris"}|<think>check</think>answer|final|'
    assert "question" not in text and "tool-result" not in text and "follow-up" not in text
    assert record == original
    value = conversation_datum(tokens, weights)
    assert value["model_input"]["chunks"][0]["tokens"] == tokens[:-1]
    assert value["loss_fn_inputs"]["target_tokens"]["data"] == tokens[1:]
    assert value["loss_fn_inputs"]["weights"]["data"] == weights[1:]
    assert not {"advantages", "logprobs"}.intersection(value["loss_fn_inputs"])


def test_generation_mask_and_exact_prefix_fallback_agree(tokenizer):
    record = conversation()
    native = encode(record, tokenizer)
    tokenizer.chat_template = tokenizer.chat_template.replace("{% generation %}", "").replace(
        "{% endgeneration %}", ""
    )
    assert encode(record, tokenizer) == native


def test_message_weights_mask_captured_history(tokenizer):
    record = conversation()
    record["messages"][2]["weight"] = 0
    record["messages"][4]["weight"] = 0
    record["messages"][-1]["weight"] = 0.5
    tokens, weights = encode(record, tokenizer)
    assert supervised_text(tokenizer, tokens, weights) == "final|"
    assert set(weights) == {0, 0.5}


def test_reasoning_cannot_be_silently_dropped(tokenizer):
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m.role + ':' + (m.content or '') }}{% endfor %}"
    )
    with pytest.raises(ValueError, match="drops reasoning"):
        encode(conversation(), tokenizer)


def test_rejects_template_that_rewrites_previous_turns(tokenizer):
    tokenizer.chat_template = "{% for m in messages %}{{ m.role + ':' + (m.content or '') }}{% endfor %}{% if add_generation_prompt %}wrong-prefix{% endif %}"
    with pytest.raises(ValueError, match="preserve the generation prefix"):
        encode(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "answer"},
                ]
            },
            tokenizer,
        )


def test_boundary_overrides_rejected():
    record = conversation()
    for key in ("tokenize", "return_assistant_tokens_mask", "truncation"):
        with pytest.raises(ValueError, match="message boundaries"):
            validate_record({**record, "chat_template_kwargs": {key: True}})


def test_nullable_calls_and_reasoning_aliases_do_not_mutate_source():
    message = {"role": "assistant", "content": "answer", "reasoning": "plan", "tool_calls": None}
    rendered = _render_messages([message])[0]
    assert rendered["reasoning_content"] == rendered["reasoning"] == "plan"
    assert "tool_calls" not in rendered
    assert message["tool_calls"] is None and "reasoning_content" not in message


def test_tool_calls_cannot_be_silently_dropped(tokenizer):
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m.role + ':' + (m.content or '') }}{% endfor %}"
    )
    record = conversation()
    for message in record["messages"]:
        message.pop("reasoning_content", None)
        message.pop("reasoning", None)
    with pytest.raises(ValueError, match="drops tool calls"):
        encode(record, tokenizer)


@pytest.mark.parametrize("provider", ["tinker", "spindle", "fireworks", "baseten"])
def test_full_conversation_is_one_native_datum(provider, tokenizer):
    modules = {
        "tinker": TinkerDatumAdapter,
        "spindle": SpindleDatumAdapter,
        "fireworks": FireworksDatumAdapter,
        "baseten": BasetenDatumAdapter,
    }
    tokens, weights = encode(conversation(), tokenizer)
    datums = list(modules[provider]().encode(conversation(), tokenizer))
    assert len(datums) == 1
    value = datum_mapping(datums[0])
    assert value["model_input"]["chunks"][0]["tokens"] == tokens[:-1]
    assert value["loss_fn_inputs"]["weights"]["data"] == weights[1:]


def test_native_masks_exclude_forced_generation_prefix_and_post_eos_padding(tokenizer):
    tokenizer.eos_token = "|"
    tokenizer.chat_template = """{% for m in messages %}{{ m.role + ':' }}{% if m.role == 'assistant' %}{% generation %}{{ 'think:' + m.content + '|\n' }}{% endgeneration %}{% else %}{{ m.content + '|' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant:think:' }}{% endif %}"""
    record = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    tokens, weights = encode(record, tokenizer)
    assert supervised_text(tokenizer, tokens, weights) == "answer|"
    prompt = tokenizer.apply_chat_template(
        record["messages"][:1], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    assert not any(weights[: len(prompt)])
    assert weights[-1] == 0
