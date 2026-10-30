# Harvey LAB → Baseten Loops

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Run the upstream `lab_core.harness.run` module with its Baseten adapter, import
native `results/<run_id>/` artifacts with Trains, train through Loops, deploy, then
collect again. The upstream task data is a Git submodule; the harness comes from
the official release wheel.
The example uses group-centered rewards and clipped PPO for one RL iteration;
see the [shared RL contract](../README.md#reinforcement-learning).

## Configure

Run commands from this directory. `trains-sdk[harvey,baseten]` installs the pinned
official `lab-core` release wheel and Loops. The `data/` submodule pins the
[Harvey LAB task documents](https://github.com/harveyai/harvey-labs/tree/v1.3.0/tasks)
to the matching v1.3.0 release. Set `LAB_ROOT` to this checkout, which contains `tasks/`.

Copy [`.env.example`](.env.example) once and edit `.env` for your services.

```bash
git submodule update --init --recursive data
uv sync --locked
cp .env.example .env
# Fill in the required credentials in .env, then load it:
set -a
. ./.env
set +a
```

Install Podman and host Pandoc as described in the upstream
[setup guide](https://github.com/harveyai/harvey-labs/blob/v1.3.0/README.md).
The upstream grader's default judge pair requires its own provider credentials;
keep those models, keys, and endpoints fixed across both rounds. Its supported
`--judges MODEL [MODEL]` option selects a different profile when desired.

For local training/vLLM, configure [train](../../train/README.md) and start from
the checkout root:

```bash
docker compose -f compose.train.yml --profile training --profile vlm up -d
```

For hosted execution use a supported Loops base model, its exact tokenizer,
your credentials, and the actual training and baseline inference URLs.

Use a non-reasoning policy: stock LAB transcripts omit hidden reasoning.
Keep the matching harness version, task documents, and `LAB_ROOT` during import.
Collect at temperature 1 with top_p 1, top_k disabled, and min_p 0. Training keeps
failed attempts and requires within-task reward variation. Use `--score-file`
when multiple judge outputs are present.

## Publish the initial policy

Fresh policies default to `LiquidAI/LFM3-Omni-2B`; use `--base-model` to override it.

Save the initial trainer state and publish the identical weights before collecting:

```bash
uv run python run.py initialize --base-model "$BASE_MODEL" \
  --training-url "$TRAINING_URL"
uv run python run.py deploy var/initial-checkpoint.json \
  --control-url http://127.0.0.1:8382 --local \
  --model-name harvey-initial --instance-type local --hf-secret unused-local \
  --output var/initial.json
INITIAL_URL=$(uv run python -c 'import json; print(json.load(open("var/initial.json"))["inference_url"])')
INITIAL_MODEL=$(uv run python -c 'import json; print(json.load(open("var/initial.json"))["model"])')
```

## Collect, grade, and train

Collect independent attempts at the same task. Each native run is a rollout;
all graded attempts, including failures, participate in its group baseline.

```bash
BASETEN_BASE_URL="$INITIAL_URL" uv run python -m lab_core.harness.run \
  --model "baseten/$INITIAL_MODEL" --task "$TASK" --temperature 1 --run-id before-1
uv run python -m lab_core.evaluation.run_eval --task "$TASK" --run-id before-1
BASETEN_BASE_URL="$INITIAL_URL" uv run python -m lab_core.harness.run \
  --model "baseten/$INITIAL_MODEL" --task "$TASK" --temperature 1 --run-id before-2
uv run python -m lab_core.evaluation.run_eval --task "$TASK" --run-id before-2
uv run python run.py prepare "$LAB_ROOT/results/before-1" "$LAB_ROOT/results/before-2" \
  --output var/chats-before
uv run python run.py train var/chats-before --behavior var/initial.json \
  --training-url "$TRAINING_URL" --tokenizer-model "$TOKENIZER_MODEL" \
  --output var/checkpoint.json
```

## Deploy

```bash
uv run python run.py deploy var/checkpoint.json \
  --control-url http://127.0.0.1:8382 --local \
  --model-name harvey-policy --instance-type local --hf-secret unused-local
```

## Collect and grade again with the deployed policy

```bash
POLICY_URL=$(uv run python -c 'import json; print(json.load(open("var/deployment.json"))["inference_url"])')
POLICY_MODEL=$(uv run python -c 'import json; print(json.load(open("var/deployment.json"))["model"])')
BASETEN_BASE_URL="$POLICY_URL" uv run python -m lab_core.harness.run \
  --model "baseten/$POLICY_MODEL" --task "$TASK" --temperature 1 --run-id after-1
uv run python -m lab_core.evaluation.run_eval --task "$TASK" --run-id after-1
BASETEN_BASE_URL="$POLICY_URL" uv run python -m lab_core.harness.run \
  --model "baseten/$POLICY_MODEL" --task "$TASK" --temperature 1 --run-id after-2
uv run python -m lab_core.evaluation.run_eval --task "$TASK" --run-id after-2
uv run python run.py prepare "$LAB_ROOT/results/after-1" "$LAB_ROOT/results/after-2" \
  --output var/chats-after
```

For hosted deployment, use the management API URL and valid instance type/HF
secret; omit `--local`. Use `--inference-model` if the served model name differs.
Keep `var/deployment.json` for the next update's `--behavior`. Change `--task` for
other LAB tasks; keep judges fixed and use held-out tasks for quality comparisons.

## SDK compatibility

The lockfile pins Loops and the Baseten management SDK together. Its local uv
override relaxes BDN's older metadata cap; training and deployment share one environment.

```bash
uv run pytest ../../trains/tests/datasets/test_harvey.py -q
```
