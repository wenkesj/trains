import copy

import pytest
from recipes.models import restore_liquid_generation_prefix
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

from trains.formats.records import validate_record
from trains.formats.tokenization import encode


def captured_call():
    return {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "old</think>action", "weight": 0},
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "new</think>action"},
        ],
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
    }


def test_restore_prefill_preserves_generated_content_history_and_source():
    record = captured_call()
    original = copy.deepcopy(record)
    restored = restore_liquid_generation_prefix(record)
    assert restored["messages"][-1] == {**original["messages"][-1], "reasoning_content": ""}
    assert restored["messages"][:-1] == original["messages"][:-1]
    assert record == original
    assert restore_liquid_generation_prefix(restored) == restored


@pytest.mark.parametrize("change", ["thinking", "reasoning", "balanced", "history", "plain"])
def test_restore_does_not_infer_reasoning_or_change_other_training_modes(change):
    record = captured_call()
    if change == "thinking":
        record["chat_template_kwargs"]["enable_thinking"] = True
    elif change == "reasoning":
        record["messages"][-1]["reasoning_content"] = "native reasoning"
    elif change == "balanced":
        record["messages"][-1]["content"] = "<think>reasoning</think>action"
    elif change == "history":
        record["messages"][1]["weight"] = 1
    else:
        record["messages"][-1]["content"] = "plain action"
    assert restore_liquid_generation_prefix(record) is record


def test_restored_action_aligns_with_original_inference_prompt_and_loss_mask():

    raw = Tokenizer(
        WordLevel(
            {"[UNK]": 0, **{c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz<>/|:\n- ")}},
            unk_token="[UNK]",
        )
    )
    raw.pre_tokenizer = Split("", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    tokenizer.chat_template = """{% for m in messages %}{{ m.role + ':' }}{% if m.role == 'assistant' and (m.reasoning_content is defined or '</think>' not in m.content) %}{{ '<think></think>' }}{% endif %}{{ m.content + '|' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant:<think></think>' }}{% endif %}"""
    record = captured_call()
    with pytest.raises(ValueError, match="generation prefix"):
        encode(validate_record(record), tokenizer)
    restored = restore_liquid_generation_prefix(record)
    tokens, weights = encode(validate_record(restored), tokenizer)
    prompt = tokenizer.apply_chat_template(
        record["messages"][:-1], tokenize=True, return_dict=False, add_generation_prompt=True
    )
    assert tokens[: len(prompt)] == prompt
    assert not any(weights[: len(prompt)])
    assert all(weights[len(prompt) :])
    expected = tokenizer.encode(record["messages"][-1]["content"] + "|", add_special_tokens=False)
    assert tokens[len(prompt) :] == expected
