[Example source](https://github.com/wenkesj/trains/tree/main/examples/tracing/claude_agents) · [Native experiment datasets](#/docs/guides/tracing)

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Run Claude Agents SDK on pinned Exercism Python tasks, store OpenInference model calls
and native dataset experiments in Langfuse, then update the policy with Tinker.

## Configure

From `examples/tracing/claude_agents`, initialize the task submodule and install
the independent project:

```sh
git -C ../../.. submodule update --init --recursive examples/tracing/exercism
uv sync --locked --group dev
cp .env.example .env
```

Configure LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL. Set your
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
