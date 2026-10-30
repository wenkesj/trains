# Fine-tune Liquid Omni locally with TRL and trace held-out evaluation.

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from langfuse import get_client
from peft import LoraConfig
from recipes.models import DEFAULT_MODEL, LFM3_REVISION, load_tokenizer
from trains.datasets import chat_dataset_from_records
from trains.formats.tokenization import encode
from transformers import AutoModelForImageTextToText, TrainerCallback
from transformers.utils.import_utils import is_flash_attn_4_available
from trl import SFTConfig, SFTTrainer

DESCRIPTION = "Fine-tune Liquid Omni locally with TRL and trace held-out evaluation."


class TrainingTrace(TrainerCallback):
    """Record real step metrics and evaluate each configured training interval."""

    def __init__(self, span, evaluate):
        """Bind the active training span and held-out generation evaluator."""
        self.span = span
        self.evaluate = evaluate
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Attach observed trainer metrics to the current Langfuse span."""
        self.history.append({"step": state.global_step, **(logs or {})})
        self.span.update(metadata={"steps": self.history})

    def on_evaluate(self, args, state, control, **kwargs):
        """Score the same disjoint held-out prompts at each evaluation interval."""
        self.evaluate(f"step-{state.global_step}")


class _Evaluator:
    """Score the fixed held-out split through the current trainer's native model."""

    def __init__(self, trainer, tokenizer, langfuse, args, results):
        """Bind checkpoint evaluation to its unchanged prompts and result file."""
        self.trainer, self.tokenizer, self.langfuse = trainer, tokenizer, langfuse
        self.args, self.results = args, results

    def evaluate(self, stage):
        """Record complete held-out generation results at the current optimizer step."""
        current = self.trainer.model
        was_training = current.training
        current.eval()
        try:
            answers = [self._answer(current, a, b, stage) for a, b in self.results["heldout_pairs"]]
        finally:
            current.train(was_training)
        result = {
            "stage": stage,
            "step": self.trainer.state.global_step,
            "exact_match": sum(item["correct"] for item in answers) / len(answers),
            "answers": answers,
        }
        self.results["evaluations"].append(result)
        (self.args.output / "results.json").write_text(json.dumps(self.results, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "answers"}), flush=True)

    def _answer(self, current, a, b, stage):
        """Retain the exact native greedy answer and its unchanged exact-match grade."""
        messages = [{"role": "user", "content": f"What is {a} + {b}? Reply with only the number."}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(current.device)
        with self.langfuse.start_as_current_observation(
            name=stage,
            as_type="generation",
            model=self.args.model,
            input=messages,
        ) as generation:
            with torch.inference_mode():
                output = current.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,
                    return_dict_in_generate=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            answer = self.tokenizer.decode(
                output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            correct = answer.strip() == str(a + b)
            generation.update(output=answer)
            generation.score(name="exact_match", value=float(correct))
        return {"a": a, "b": b, "answer": answer, "correct": correct}


def main():
    """Train Omni LoRA locally and save disjoint before/during/after scores."""
    args = _arguments()
    torch.manual_seed(42)
    model, tokenizer, targets = _load_model(args)
    training_pairs, heldout_pairs = _splits(args)
    dataset, evaluation = _examples(tokenizer, training_pairs), _examples(tokenizer, heldout_pairs)
    args.output.mkdir(parents=True, exist_ok=False)
    results = {
        "model": args.model,
        "model_revision": LFM3_REVISION if args.model == DEFAULT_MODEL else None,
        "attention_backend": model.config.get_text_config()._attn_implementation,
        "train_pairs": training_pairs,
        "heldout_pairs": heldout_pairs,
        "evaluations": [],
    }
    _run_training(get_client(), args, model, tokenizer, targets, dataset, evaluation, results)


def _arguments():
    """Require the declared training schedule and a usable CUDA device."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--train-examples", type=int, default=256)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("artifacts/addition"))
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.train_examples, args.eval_examples) < 1:
        parser.error("Training and evaluation sizes must be positive")
    if not torch.cuda.is_available():
        parser.error("This Omni training example requires CUDA-enabled PyTorch and a GPU")
    return args


def _load_model(args):
    """Load pinned Omni weights with FlashAttention 4 and language-only LoRA targets."""
    tokenizer = load_tokenizer(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        **({"revision": LFM3_REVISION} if args.model == DEFAULT_MODEL else {}),
        attn_implementation="sdpa",
    )
    model.to("cuda")
    if is_flash_attn_4_available():
        model.model.language_model.set_attn_implementation("flash_attention_4")
    targets = [
        name
        for name, module in model.named_modules()
        if name.startswith("model.language_model.") and isinstance(module, torch.nn.Linear)
    ]
    if not targets:
        raise ValueError("No Liquid Omni language projection modules found")
    return model, tokenizer, targets


def _splits(args):
    """Keep the original deterministic, pair-disjoint four-digit addition split."""
    rng = random.Random(42)
    pairs = set()
    while len(pairs) < args.train_examples + args.eval_examples:
        pairs.add(tuple(sorted((rng.randrange(1000, 10000), rng.randrange(1000, 10000)))))
    pairs = sorted(pairs)
    rng.shuffle(pairs)
    training, heldout = pairs[: args.train_examples], pairs[args.train_examples :]
    assert not set(training) & set(heldout)
    return training, heldout


def _run_training(langfuse, args, model, tokenizer, targets, dataset, evaluation, results):
    """Keep native training and generation observations inside the experiment span."""
    try:
        with langfuse.start_as_current_observation(
            name="trl-liquid-addition", as_type="span"
        ) as run:
            run.update(
                input={
                    "model": args.model,
                    "steps": args.steps,
                    "training_examples": len(dataset),
                    "heldout_examples": len(evaluation),
                }
            )
            with langfuse.start_as_current_observation(
                name="local-sft", as_type="span"
            ) as training:
                _train(
                    args,
                    model,
                    tokenizer,
                    targets,
                    dataset,
                    evaluation,
                    results,
                    langfuse,
                    training,
                )
            run.update(
                output={
                    "evaluations": [
                        {k: v for k, v in result.items() if k != "answers"}
                        for result in results["evaluations"]
                    ],
                }
            )
    finally:
        langfuse.flush()


def _train(args, model, tokenizer, targets, dataset, evaluation, results, langfuse, training):
    """Run the full configured schedule and persist the native adapter and metrics."""

    def evaluate(stage):
        """Resolve the current trainer after its native initialization."""
        evaluator.evaluate(stage)

    trace = TrainingTrace(training, evaluate)
    trainer = _trainer(args, model, tokenizer, targets, dataset, evaluation, [trace])
    evaluator = _Evaluator(trainer, tokenizer, langfuse, args, results)
    evaluator.evaluate("before-sft")
    trained = trainer.train()
    if not results["evaluations"] or results["evaluations"][-1]["step"] != args.steps:
        evaluator.evaluate("after-sft")
    results["training"], results["history"] = trained.metrics, trace.history
    trainer.save_model(str(args.output / "adapter"))
    tokenizer.save_pretrained(args.output / "adapter")
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    training.update(output=trained.metrics)


def _trainer(args, model, tokenizer, targets, dataset, evaluation, callbacks):
    """Configure the original native TRL optimizer, masking, and evaluation schedule."""
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=evaluation,
        callbacks=callbacks,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0,
            target_modules=targets,
            task_type="CAUSAL_LM",
        ),
        args=SFTConfig(
            output_dir=str(args.output),
            max_steps=args.steps,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=2,
            learning_rate=1e-4,
            max_length=256,
            completion_only_loss=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=args.eval_every,
            save_strategy="no",
            report_to="none",
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            seed=42,
        ),
    )


def _examples(tokenizer, pairs):
    """Validate conversations with Trains before mapping them to native TRL inputs."""
    chats = chat_dataset_from_records(
        {
            "messages": [
                {"role": "user", "content": f"What is {a} + {b}? Reply with only the number."},
                {"role": "assistant", "content": str(a + b)},
            ],
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
        }
        for a, b in pairs
    )
    return chats.map(
        lambda record: _trl_record(record, tokenizer), remove_columns=chats.column_names
    )


def _trl_record(record, tokenizer):
    """Use Trains' aligned assistant weights as TRL's unshifted completion mask."""
    tokens, weights = encode(record, tokenizer)
    return {
        "input_ids": tokens,
        "attention_mask": [1] * len(tokens),
        "completion_mask": [int(weight) for weight in weights],
    }


if __name__ == "__main__":
    main()
