import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from transformers import BatchEncoding
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

import run


class Tokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, return_dict=False, **options
    ):
        assert return_dict is False
        assert options == {"enable_thinking": False, "preserve_thinking": True}
        text = "".join(
            message["role"]
            + ":"
            + message["content"]
            + ("\n" if message["role"] != "assistant" else "")
            for message in messages
        )
        text = text + "assistant:" if add_generation_prompt else text
        return [ord(character) for character in text] if tokenize else text

    def __call__(self, text, *, return_tensors):
        assert return_tensors == "pt"
        return BatchEncoding({"input_ids": torch.tensor([[ord(c) for c in text]])})

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens
        return "".join(chr(int(token)) for token in tokens)


class Model:
    device = "cpu"
    training = True

    def eval(self):
        self.training = False

    def train(self, mode):
        self.training = mode

    def generate(self, *, input_ids, **options):
        assert not self.training and not torch.is_grad_enabled()
        assert options == {
            "max_new_tokens": 16,
            "do_sample": False,
            "return_dict_in_generate": False,
            "pad_token_id": 0,
        }
        return torch.cat([input_ids, torch.tensor([[ord(c) for c in "2000"]])], dim=1)


def test_training_and_heldout_pairs_are_deterministic_and_disjoint():
    args = SimpleNamespace(train_examples=256, eval_examples=64)
    train, heldout = run._splits(args)
    assert (train, heldout) == run._splits(args)
    assert len(set(train)) == 256 and len(set(heldout)) == 64
    assert not set(train) & set(heldout)
    assert all(1000 <= a <= b < 10000 for a, b in train + heldout)


def test_native_completion_mask_keeps_the_entire_prompt_out_of_the_loss():
    tokenizer = Tokenizer()
    (record,) = run._examples(tokenizer, [(1000, 1000)])
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 1000 + 1000? Reply with only the number."}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        preserve_thinking=True,
    )
    assert record["input_ids"][: len(prompt)] == prompt
    assert record["completion_mask"][: len(prompt)] == [0] * len(prompt)
    assert record["completion_mask"][len(prompt) :] == [1] * (
        len(record["input_ids"]) - len(prompt)
    )
    assert record["attention_mask"] == [1] * len(record["input_ids"])


def test_trains_conversion_drives_trl_labels(monkeypatch):
    chats = Mock(wraps=run.chat_dataset_from_records)
    encode = Mock(wraps=run.encode)
    monkeypatch.setattr(run, "chat_dataset_from_records", chats)
    monkeypatch.setattr(run, "encode", encode)
    dataset = run._examples(Tokenizer(), [(1000, 1000), (1000, 1001)])
    chats.assert_called_once()
    assert encode.call_count == 2
    collator = DataCollatorForLanguageModeling(pad_token_id=0, completion_only_loss=True)
    batch = collator(list(dataset))
    for index, record in enumerate(dataset):
        expected = [
            token if weight else -100
            for token, weight in zip(record["input_ids"], record["completion_mask"], strict=True)
        ]
        assert batch["labels"][index, :len(expected)].tolist() == expected


def test_trains_masks_template_suffix_after_eos():
    class TerminatedTokenizer(Tokenizer):
        eos_token_id = ord("~")

        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            if messages[-1]["role"] == "assistant":
                return result + [self.eos_token_id, ord("\n")]
            return result

    (record,) = run._examples(TerminatedTokenizer(), [(1000, 1000)])
    assert record["input_ids"][-2:] == [ord("~"), ord("\n")]
    assert record["completion_mask"][-2:] == [1, 0]


def test_evaluation_retains_wrong_answers_scores_each_attempt_and_restores_training(tmp_path):
    model = Model()
    trainer = SimpleNamespace(model=model, state=SimpleNamespace(global_step=100))
    generations = []

    def observation(**kwargs):
        assert kwargs["as_type"] == "generation" and kwargs["name"] == "step-100"
        generation = Mock()
        generations.append(generation)
        return nullcontext(generation)

    results = {"heldout_pairs": [(1000, 1000), (1000, 1001)], "evaluations": []}
    args = SimpleNamespace(model=run.DEFAULT_MODEL, output=tmp_path)
    evaluator = run._Evaluator(
        trainer,
        Tokenizer(),
        SimpleNamespace(start_as_current_observation=observation),
        args,
        results,
    )
    evaluator.evaluate("step-100")
    assert model.training is True
    saved = json.loads((tmp_path / "results.json").read_text())
    (evaluation,) = saved["evaluations"]
    assert evaluation["stage"] == "step-100" and evaluation["step"] == 100
    assert evaluation["exact_match"] == 0.5
    assert [answer["correct"] for answer in evaluation["answers"]] == [True, False]
    assert [answer["answer"] for answer in evaluation["answers"]] == ["2000", "2000"]
    generations[0].score.assert_called_once_with(name="exact_match", value=1.0)
    generations[1].score.assert_called_once_with(name="exact_match", value=0.0)
