# Hub OpenHands trajectories → Liquid → Pi → PPO

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

[Example source](https://github.com/wenkesj/trains/tree/main/examples/hf-agent-tracing) · [Session adapters](https://wenkesj.github.io/trains/#/docs/guides/sessions)

Fine-tune `LiquidAI/LFM3-Omni-2B` on
[NVIDIA SWE-Hero OpenHands trajectories](https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories),
then collect new Pi coding attempts and train from their native sessions.
Dataset and model revisions are pinned. Liquid's tokenizer renders conversations;
Loops performs training and token sampling.

The demonstrations are **CC BY 4.0**, attributed to NVIDIA. Source repositories
use MIT, Apache-2.0, or BSD licenses; preparation rejects other labels and retains
repository terms and attribution in each chat, datum, and `artifacts/source.json`.
Preserve those notices when sharing data. The loader restores omitted tool-result
IDs only when exactly one call is outstanding; ambiguous matches fail.

## Setup

```sh
cd examples/hf-agent-tracing
git -C ../.. submodule update --init --recursive examples/tracing/exercism
uv sync --python 3.12
docker build -t trains-pi-harness:0.73.1 .
cp .env.example .env
# Configure the Loops endpoint, then load it:
set -a; source .env; set +a
```

`--attempts` sets repetitions per exercise. Keep a disjoint held-out task set,
for example `--exercises isogram,word-count`, for before/after evaluation. The
shared Git submodule pins [Exercism Python](https://github.com/exercism/python);
each receipt records the revision and task-file checksum. `--max-calls` and
`--max-tokens` adjust collection budgets.

Docker is required for collection and grading. Preparation streams selected trajectories and downloads the tokenizer.

## Supervised training

```sh
uv run python run.py prepare --limit 4
uv run python run.py sft --steps 12
```

Preparation keeps complete OpenHands conversations, including tool calls and
results. `BasetenDatumAdapter` creates supervised datums using Liquid's native
template. `artifacts/source.json` records selected trajectory IDs, source
repositories, licenses, attribution, and revisions. Training saves optimizer and
sampler checkpoints plus actual metrics in `policy.json`. The configured trainer
must support the longest prepared conversation; demonstrations can be long.

## Isolated coding attempts

```sh
uv run python run.py collect --exercises anagram,binary-search,matrix --attempts 4
```

Each attempt runs the pinned native Pi CLI in a fresh container, with only its
workspace, private profile, and native session directory mounted. Pi reads
upstream Exercism instructions and edits starter solutions. Extensions, skills,
prompt templates, and context discovery are disabled. Provider credentials stay
in the host bridge. Host networking lets the agent reach that bridge.

A separate container runs hidden upstream tests with networking disabled and a
read-only candidate mount. Both containers have resource limits, read-only root
filesystems, and no added capabilities. The agent has a ten-minute timeout and
20-model-call budget with 2,048 output tokens per call. The verifier uses Python
3.12 and resource-bounded, unchanged upstream tests in a separate directory. Failures remain in the batch.

Native sessions default to `~/.pi/agent/sessions/trains-hf-agent-tracing/`;
`--sessions` changes this root. The printed collection directory retains
workspaces, logs, grades, and exact prompt/action token receipts. Sampling uses
temperature 1, top-p 1, and no top-k truncation against one frozen checkpoint.

## PPO from native sessions

```sh
uv run python run.py rl --collection artifacts/collection-... --steps 1
```

`chat_dataset_from_sessions(..., adapter=PiSessionAdapter())` imports the new
sessions. Every assistant action must match its sampler receipt. Missing or
changed actions and mixed checkpoints reject the whole batch before updating.
Datums retain raw sampled tokens, including reasoning and terminal tokens.

The first PPO round restores the SFT weights with fresh Adam state; later rounds
resume the PPO optimizer. PPO uses captured native sampler log probabilities
when available and centers rewards over all attempts, including failures. A complete batch without sampler probabilities uses frozen
teacher-forced references; metrics identify the reference source. A batch with
no reward variation stops without an update. Updated checkpoints and metrics
are saved in the collection directory. The command updates `artifacts/policy.json`
so the next `collect` uses the trained checkpoint. `--learning-rate` tunes SFT or PPO. No train/serve drift correction is
claimed for teacher-forced references.

## Verification

```sh
uv run --group dev pytest tests -q
```

The integration test runs containerized Pi with scripted sampler outputs,
verifies upstream Exercism grading, passing and failing candidates, imports
native sessions, and checks
PPO replay and tamper rejection with a recording trainer. Live training quality
must be evaluated separately with the configured Loops service.
