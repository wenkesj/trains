[Example source](https://github.com/wenkesj/trains/tree/main/examples/tracing/deepagents) · [Native experiment datasets](#/docs/guides/tracing)

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Run DeepAgents on pinned Exercism Python tasks, store OpenInference model calls
and native dataset experiments in LangSmith, then update the policy with Spindle.

## Configure

From `examples/tracing/deepagents`, initialize the task submodule and install
the independent project:

```sh
git -C ../../.. submodule update --init --recursive examples/tracing/exercism
uv sync --locked --group dev
cp .env.example .env
```

Configure LANGSMITH_API_KEY and optional LANGSMITH_WORKSPACE_ID. Set your
`TINKER_API_KEY`, `TINKER_BASE_URL`, and `INFERENCE_URL` for the training provider
and checkpoint-aware inference gateway. Use a supported model and matching renderer.
Load `.env` before running the commands below.

## Evaluate, update, evaluate

```sh
uv run python run.py init-policy var/initial-policy.json \
  --training-url "$TINKER_BASE_URL" --inference-url "$INFERENCE_URL" \
  --base-model "$BASE_MODEL" --renderer "$RENDERER"
uv run python run.py run --policy var/initial-policy.json --output var/run-1
```

The run evaluates the initial sampler, collects repeated graded attempts,
exports experiment-linked model calls, applies one PPO update, publishes a new
sampler, and evaluates it on the same held-out exercises.

Candidate containers receive starter files and instructions. A separate verifier
runs the original tests; test pass fractions supply rewards. Retain failed attempts
and use harder tasks or more repetitions if every reward is equal.

Fresh output directories contain native experiment receipts, candidate files,
HF chats, checkpoints, metrics, and `report.json`. Override task splits with
`--train` and `--evaluation`, keeping them disjoint.

Spindle uses its Tinker-compatible training contract. Its inference gateway is
configured separately; publication receipts pin the sampler used for collection.
