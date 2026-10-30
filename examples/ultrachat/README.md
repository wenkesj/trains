# UltraChat → Fireworks

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Train on the `messages` column of
[HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k),
pinned to a dataset revision. Each full conversation becomes native Fireworks
training data with assistant-only loss weights. No reward or tracing system is
needed. Start with a small slice; scale batching and context limits for larger runs.

From this directory:

Copy [`.env.example`](.env.example) once and edit `.env` for your services.

```bash
uv sync --locked
cp .env.example .env
# Fill in the required credentials in .env, then load it:
set -a
. ./.env
set +a
uv run python run.py train \
  --base-model YOUR_TRAINABLE_FIREWORKS_MODEL \
  --tokenizer-model YOUR_MATCHING_HF_TOKENIZER --samples 32 --steps 1
```

Choose a supported Fireworks training model and its exact tokenizer. The script
uses native `ServiceClient`, `forward_backward`, `optim_step`, `save_state`, and
`save_weights_for_sampler`; it saves both checkpoint paths to `var/checkpoint.json`.
The script uses one small batch; lower `--samples` if full conversations exceed
your context or memory budget.

For local training, start the root service following [train](../../train/README.md):

```bash
# From the checkout root:
docker compose -f compose.train.yml --profile training --profile vlm up -d
# From this example:
FIREWORKS_API_KEY=local uv run python run.py train \
  --training-url http://127.0.0.1:8382 \
  --base-model LiquidAI/LFM3-Omni-2B --tokenizer-model LiquidAI/LFM3-Omni-2B \
  --samples 2
```

The local service accepts the saved sampler path as the model in Chat Completions
at `http://127.0.0.1:8382/v1`; it serves the selected checkpoint through the configured inference backend.

## Cache the dataset

```sh
uv run python run.py cache
uv run python run.py train --offline --samples 8 --evaluation-samples 8 --steps 3
```

The pinned training and test splits are saved under `var/data/ultrachat/` and reused
without HF requests. Training fills this cache automatically on the first run.
Use `--data-dir PATH` on either command to choose another location. `--offline`
requires both the selected dataset splits and tokenizer to be cached already.

## Measure training progress

Add `--steps 3 --evaluation-samples 16` to record held-out assistant-token loss
before training and after each update. Evaluation uses `test_sft`; training uses
`train_sft`. Lower held-out loss indicates better likelihood on those conversations.
The checkpoint receipt records the dataset revision, slices, token counts, and losses.

Use `--checkpoint var/checkpoint.json` with a new `--output` path to resume weights
and optimizer state. `--offset` selects another training slice; keep the evaluation
slice fixed across rounds. Conversations are retained in full.

## Deploy and call the updated model

Deployment is a separate native SDK step; saving sampler weights alone is not a
serving deployment. For the local proxy:

```bash
FIREWORKS_API_KEY=local uv run python run.py deploy var/checkpoint.json \
  --control-url http://127.0.0.1:8382 --account-id local \
  --model-id ultrachat-policy --deployment-id ultrachat-serving
```

For hosted Fireworks omit `--control-url`, select your account, and supply
`--job-id YOUR_TRAINER_JOB --checkpoint-name YOUR_CHECKPOINT_RESOURCE` when those
cannot be derived from the receipt. The script promotes the exact saved
checkpoint, creates the deployment with the native Fireworks SDK, waits for
READY, and writes `var/deployment.json`. Call it with the native inference SDK:

```bash
uv run python - <<'PYTHON'
import json, os
from fireworks import Fireworks
receipt = json.load(open("var/deployment.json"))
with Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], base_url=receipt["inference_url"]) as client:
    response = client.chat.completions.create(
        model=receipt["model"], messages=[{"role": "user", "content": "Explain supervised fine-tuning briefly."}],
        max_tokens=128,
    )
    print(response.choices[0].message.content)
PYTHON
```

Use unique model/deployment IDs for each run. Root training configuration owns
GPU and inference management; this example has no additional Compose services. Hosted
serving capacity persists until deleted through the native deployment SDK.
