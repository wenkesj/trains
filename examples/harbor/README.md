# Harbor → Tinker / Spindle

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Use the official Harbor CLI and Terminus-2 agent on `terminal-bench@latest` with
Docker. Import the native job with Trains, train through the Tinker SDK, publish
sampler weights, then collect again with the deployed policy. There is
no custom task, agent, or verifier. Training uses group-centered rewards and
clipped PPO, following the [shared RL contract](../README.md#reinforcement-learning).

## Configure

Run commands from this directory. Start the local service following
[train](../../train/README.md); from the checkout root:

Copy [`.env.example`](.env.example) once and edit `.env` for your services.

```bash
docker compose -f compose.train.yml --profile training --profile vlm up -d
```

Docker must be running; Harbor downloads task images and runs their verifiers.
For the local API:

```bash
uv sync --locked
cp .env.example .env
# Fill in the required credentials in .env, then load it:
set -a
. ./.env
set +a
```

Use the exact tokenizer and renderer for the served policy. Keep complete reasoning
and histories; the logging flags below disable summarization. Collect at temperature 1
with top_p 1, top_k disabled, and min_p 0. Use `--reward-name` for multiple metrics.
An entirely flat reward batch needs more or harder attempts.

## Publish the initial policy

Fresh policies default to `LiquidAI/LFM3-Omni-2B`; use `--base-model` to override it.

Tinker inference uses sampler URIs rather than bare base-model IDs. Publish an
untrained LoRA checkpoint first (the local API also accepts base-model IDs):

```bash
uv run python run.py deploy --base-model "$BASE_MODEL" \
  --training-url "$TRAINING_URL" --inference-url "$INFERENCE_URL" \
  --output var/initial.json
INITIAL_MODEL=$(uv run python -c 'import json; print(json.load(open("var/initial.json"))["model"])')
```

## Collect and train

```bash
uv run harbor run -d terminal-bench@latest -a terminus-2 \
  -m "openai/$INITIAL_MODEL" --base-url "$INFERENCE_URL" \
  -e docker -l 2 -k 4 --jobs-dir var/jobs --job-name before-update \
  --ak temperature=1 --ak store_all_messages=true --ak enable_summarize=false \
  --ak proactive_summarization_threshold=0 --ak interleaved_thinking=true
uv run python run.py prepare var/jobs/before-update --output var/chats-before
uv run python run.py train var/chats-before \
  --training-url "$TRAINING_URL" --behavior var/initial.json \
  --tokenizer-model "$TOKENIZER_MODEL" --output var/checkpoint.json
```

## Publish the trained policy

```bash
uv run python run.py deploy var/checkpoint.json \
  --inference-url "$INFERENCE_URL"
```

## Collect again with the deployed policy

Read the callable endpoint and served-model name from the deployment receipt:

```bash
POLICY_URL=$(uv run python -c 'import json; print(json.load(open("var/deployment.json"))["inference_url"])')
POLICY_MODEL=$(uv run python -c 'import json; print(json.load(open("var/deployment.json"))["model"])')
OPENAI_API_KEY="$TINKER_API_KEY" uv run harbor run \
  -d terminal-bench@latest -a terminus-2 \
  -m "openai/$POLICY_MODEL" --base-url "$POLICY_URL" \
  -e docker -l 2 -k 4 --jobs-dir var/jobs --job-name after-update \
  --ak temperature=1 --ak store_all_messages=true --ak enable_summarize=false \
  --ak proactive_summarization_threshold=0 --ak interleaved_thinking=true
uv run python run.py prepare var/jobs/after-update --output var/chats-after
```

## Other benchmarks and environments

Change `-d terminal-bench@latest` to another registered dataset, or use `-p PATH`
for existing tasks. Keep the logging flags and use the same importer. Other agents
work when they save full messages or exact captured ATIF calls. Change `-e docker`
to `-e modal` and configure Harbor's Modal credentials to use Modal environments.
No additional Compose services are required.

## Hosted Tinker and Modal Spindle

For hosted Tinker, use provider credentials, a supported model, and the SDK's
training URL. Set `INFERENCE_URL` to:

```text
https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1
```

For Spindle, set `TINKER_BASE_URL` and `TINKER_API_KEY` (`tml-` prefix with this SDK).
Pass `--provider spindle` on `train` and initial `deploy`, and use the Spindle
training URL. Supply a checkpoint-aware chat gateway through `--inference-url`;
Spindle's training URL supplies native sampling, not a general chat endpoint.
The local trainer supplies both contracts.

`deploy` publishes sampler weights and verifies native sampling registration.
Read `model` and `inference_url` from its receipt. Keep that receipt as `--behavior`
for the next training round. Compare held-out tasks with unchanged verifiers.
