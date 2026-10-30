# Tracing examples

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Collect real coding attempts, store OpenInference spans and graded experiments in
a tracing provider, then train and evaluate a new policy.

| Example | Harness | Trace provider | Training |
| --- | --- | --- | --- |
| [Local TRL](trl/README.md) | TRL SFTTrainer | Langfuse | Local PyTorch |
| [OpenAI Agents](openai_agents/README.md) | OpenAI Agents SDK | Phoenix | Tinker |
| [Claude Agents](claude_agents/README.md) | Claude Agent SDK | Langfuse | Tinker |
| [Codex](codex/README.md) | Codex CLI | Arize AX | Tinker |
| [OpenCode](opencode/README.md) | OpenCode CLI | LangSmith | Tinker |
| [Deep Agents](deepagents/README.md) | Deep Agents | LangSmith | Spindle |

Each directory is an independent uv project with its own code, dependencies,
lockfile, tests, and run instructions. Projects depend on the local Trains library
and [recipes.rl](../recipes/README.md). Start with the README for your harness.

The shared Exercism task source is the `exercism/` Git submodule. Initialize it
with `git submodule update --init --recursive examples/tracing/exercism` from the
checkout root. Runs record the revision and file checksums; outputs stay under
each project's `var/`.
