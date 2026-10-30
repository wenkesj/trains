[Example source](https://github.com/wenkesj/trains/tree/main/examples/harvey_lab) · [Dataset import](#/docs/guides/benchmarks)

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Use the native Harvey harness, preserve its graded artifacts, and prepare a Trains
dataset for the Baseten SDK. The project lives in `examples/harvey_lab`.

## Collect

Initialize the pinned `data/` submodule and set `LAB_ROOT` to that checkout.
Use the matching upstream LAB wheel and a non-reasoning policy; stock transcripts
omit hidden reasoning. Keep judge models and credentials fixed.

```sh
uv run python -m lab_core.harness.run \
  --model "baseten/$POLICY_MODEL" --task "$TASK" --temperature 1 --run-id before-1
uv run python -m lab_core.evaluation.run_eval --task "$TASK" --run-id before-1
```

Repeat independently with `--run-id before-2` before preparing the dataset.

## Prepare and update

After configuring your provider credentials and endpoints:

```sh
uv sync --locked
uv run python run.py prepare data/results/before-1 data/results/before-2 --output var/chats
uv run python run.py train var/chats --behavior var/initial.json \
  --training-url "$TRAINING_URL" --tokenizer-model "$TOKENIZER_MODEL" \
  --output var/checkpoint.json
uv run python run.py deploy var/checkpoint.json --control-url "$BASETEN_CONTROL_URL" \
  --model-name harvey-policy --instance-type "$INSTANCE_TYPE" --hf-secret "$HF_SECRET"
```

`var/initial.json` must be the publication receipt for the collection policy.
Use the updated deployment receipt for the next collection. Keep all graded
attempts; measure quality with held-out tasks and unchanged graders.
