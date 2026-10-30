# Inspect → Tinker / Spindle

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Collect native Inspect `.eval` logs, convert their model calls into portable
conversations, train with PPO, and evaluate the published policy.
[ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) provides real chart
images and answers. Training and evaluation use separate dataset splits. Numeric
answers allow 5% relative error; other answers use case-insensitive exact matching.
No hosted judge is required.

## Configure

Start the [local services](../../train/README.md) from the checkout root, then run
these commands from this directory:

```sh
uv sync --locked
cp .env.example .env
set -a
. ./.env
set +a
```

The default `LiquidAI/LFM3-Omni-2B` uses its native tokenizer and image processor.
For another model, supply its exact tokenizer and `--renderer`. Keep raw API
logging, complete responses, and untruncated sampling for PPO replay.

## Publish the initial policy

Fresh policies default to `LiquidAI/LFM3-Omni-2B`; use `--base-model` to override it.

```sh
uv run python run.py deploy --base-model "$BASE_MODEL" \
  --training-url "$TRAINING_URL" --inference-url "$INFERENCE_URL" \
  --output var/initial.json
INITIAL_MODEL=$(uv run python -c 'import json; print(json.load(open("var/initial.json"))["model"])')
```

## Evaluate and collect

Record a held-out baseline, then collect repeated attempts on the training split:

```sh
uv run inspect eval task.py@chartqa -T split=test -T limit=4 \
  --model "openai-api/tinker/$INITIAL_MODEL" --model-base-url "$INFERENCE_URL" \
  --epochs 4 --temperature 1 --top-p 1 --log-model-api --log-dir var/baseline
uv run inspect eval task.py@chartqa -T split=train -T limit=4 \
  --model "openai-api/tinker/$INITIAL_MODEL" --model-base-url "$INFERENCE_URL" \
  --epochs 4 --temperature 1 --top-p 1 --log-model-api --log-dir var/train
```

Substitute the training log path printed by Inspect:

```sh
uv run python run.py prepare var/train/YOUR_RUN.eval --output var/chats
uv run python run.py train var/chats --behavior var/initial.json \
  --training-url "$TRAINING_URL" --tokenizer-model "$TOKENIZER_MODEL" \
  --output var/checkpoint.json
uv run python run.py deploy var/checkpoint.json --inference-url "$INFERENCE_URL"
```

PPO requires within-question reward variation. Increase attempts or select another
training slice with `-T offset=...` if all attempts receive the same reward.

## Evaluate the updated policy

```sh
POLICY_MODEL=$(uv run python -c 'import json; print(json.load(open("var/deployment.json"))["model"])')
uv run inspect eval task.py@chartqa -T split=test -T limit=4 \
  --model "openai-api/tinker/$POLICY_MODEL" --model-base-url "$INFERENCE_URL" \
  --epochs 4 --temperature 1 --top-p 1 --log-model-api --log-dir var/after
```

Repeat collection on `split=train`, using `var/deployment.json` as the next
`--behavior` receipt. Keep the held-out split and scoring fixed.

## Other tasks and providers

`task.py@chartography` retains the upstream Chartography task for datasets you can
access. Its default judge is the fixed, untuned Omni model configured through
`JUDGE_MODEL`, `JUDGE_BASE_URL`, and `JUDGE_API_KEY`. Keep that judge independent of
the trained sampler. Other Inspect tasks use the same import/train/deploy flow.
Use `--scorer` when a log contains multiple scores; numeric and C/I scores work.

For hosted Tinker, use its training endpoint and checkpoint-aware inference gateway.
For Modal Spindle, set `TINKER_BASE_URL` and `TINKER_API_KEY`, and pass
`--provider spindle` to `train` and initial `deploy`. Supply an image-capable chat
gateway through `--inference-url`; the training endpoint alone does not serve chat.

Preparation selects the evaluated model, excludes judge calls and incomplete
samples, and embeds images. Local and embedded media work directly; remote media
needs a resolver. See the [shared RL contract](../README.md#reinforcement-learning).

Check with `uv run pytest ../../trains/tests/datasets/test_inspect.py tests -q`.
