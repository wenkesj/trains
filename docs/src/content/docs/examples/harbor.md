[Example source](https://github.com/wenkesj/trains/tree/main/examples/harbor) · [Dataset import](#/docs/guides/benchmarks)

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Use the native Harbor harness, preserve its graded artifacts, and prepare a Trains
dataset for the Tinker SDK. The project lives in `examples/harbor`.

## Collect

Collect with the official Harbor CLI and a registered task dataset. Keep full
messages, disable summarization, and retain reasoning. Collect repeated attempts
with fixed verifiers and your published sampler as the model.

```sh
uv run harbor run -d terminal-bench@latest -a terminus-2 \
  -m "openai/$POLICY_MODEL" --base-url "$INFERENCE_URL" \
  -e docker -l 2 -k 4 --jobs-dir var/jobs --job-name finished-job \
  --ak temperature=1 --ak store_all_messages=true --ak enable_summarize=false \
  --ak proactive_summarization_threshold=0 --ak interleaved_thinking=true
```

## Prepare and update

After configuring your provider credentials and endpoints:

```sh
uv sync --locked
uv run python run.py prepare var/jobs/finished-job --output var/chats
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
