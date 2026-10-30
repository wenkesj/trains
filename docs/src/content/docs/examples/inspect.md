[Example source](https://github.com/wenkesj/trains/tree/main/examples/inspect) · [Dataset import](#/docs/guides/benchmarks)

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Use the native Inspect harness, preserve its graded artifacts, and prepare a Trains
dataset for the Tinker SDK. The project lives in `examples/inspect`.

## Collect

Use Inspect's `--log-model-api` option to retain every raw request. The bundled
Chartography task uses a separate fixed judge. Choose a supported vision model
and its exact processor and renderer.

```sh
uv run inspect eval task.py@chartography \
  --model "openai-api/tinker/$POLICY_MODEL" --model-base-url "$INFERENCE_URL" \
  -T hf_repo=surgeai/chartography -T limit=2 -T model_family=qwen3.5 \
  -T "judge_model=$JUDGE_MODEL" --epochs 4 --temperature 1 --top-p 1 \
  --log-model-api --log-dir logs
```

Use the `.eval` path printed by Inspect when preparing the dataset.

## Prepare and update

After configuring your provider credentials and endpoints:

```sh
uv sync --locked
uv run python run.py prepare logs/evaluation.eval --output var/chats
uv run python run.py train var/chats --behavior var/initial.json \
  --training-url "$TRAINING_URL" --tokenizer-model "$TOKENIZER_MODEL" --renderer "$RENDERER" \
  --output var/checkpoint.json
uv run python run.py deploy var/checkpoint.json --inference-url "$INFERENCE_URL"
```

`var/initial.json` must be the publication receipt for the collection policy.
Use the updated deployment receipt for the next collection. Keep all graded
attempts; measure quality with held-out tasks and unchanged graders.

For Spindle, select `--provider spindle` when training and publishing the initial
policy. Supply your deployed service URL and a checkpoint-aware inference gateway.
