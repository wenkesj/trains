# Deep Agents → LangSmith → Spindle

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Solve Exercism Python tasks, grade them with upstream tests, and train from the
model calls stored in LangSmith's native datasets and experiments.

## Run

Requires Python 3.12, uv, Docker, and the [local trainer and inference server](../../../train/README.md).
Run from `examples/tracing/deepagents`:

Copy [`.env.example`](.env.example) once and edit `.env` for your services.

```sh
git -C ../../.. submodule update --init --recursive examples/tracing/exercism
uv sync --locked --group dev
cp .env.example .env
# Fill in the required credentials in .env, then load it:
set -a
. ./.env
set +a
uv run python run.py init-policy var/initial-policy.json
uv run python run.py run --policy var/initial-policy.json --output var/run-1
```

Defaults use `LiquidAI/LFM3-Omni-2B`, training at `localhost:8382`, and chat inference at
`localhost:8382/v1`. Set `TRAIN_INFERENCE_SHARED_GPU=true` when the local services share a GPU.

For hosted Spindle, set `TINKER_API_KEY` and `TINKER_BASE_URL` for your deployed
service (the pinned SDK requires a `tml-` key prefix). Pass `--inference-url` to `init-policy` with a checkpoint-aware OpenAI
chat gateway. Spindle training uses the native Tinker client and
`SpindleDatumAdapter`; its training URL alone does not provide that chat gateway.
The local trainer supports the same training contract for development.

Set `LANGSMITH_ENDPOINT` for another region and `LANGSMITH_PROJECT` for an existing
project. Agent workspaces also require `LANGSMITH_AGENT_ID` and
`LANGSMITH_AGENT_ENVIRONMENT`; use the environment's `tracer_session_name` as the project.

## What the run produces

`init-policy` saves optimizer state and publishes the collection sampler. `run`
evaluates it, collects repeated training attempts, exports OpenInference model calls,
performs PPO updates through Spindle's contract, publishes a new sampler, and
repeats the held-out evaluation.

Training defaults to `anagram,binary-search,matrix`; evaluation uses `isogram,word-count`.
Override with `--train`, `--evaluation`, `--repetitions`, or `--repository`.
Candidate containers receive starter files and instructions; a separate verifier
receives the original tests. Rewards are test pass fractions, including failures.
PPO needs within-task reward variation: use harder tasks or more attempts if a batch is flat.

Each fresh `--output` directory contains experiment receipts, candidates, test
results, HF chats, checkpoints, metrics, and `report.json`. These are integration
examples; before/after scores can regress.

Uses the upstream LangChain and OpenAI OpenInference instrumentors. Native filesystem and shell tools operate in the candidate Docker container, with a 20-call budget and delegation disabled.

Check this independent project with `uv run pytest -q`.

Use `--steps 3` for three clipped PPO updates per collection, and
`--evaluation-repetitions 4` for repeated held-out attempts. Both default to one.

LangSmith export waits for every recorded SDK call and grade before PPO.
