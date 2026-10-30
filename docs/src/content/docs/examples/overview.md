[Example source](https://github.com/wenkesj/trains/tree/main/examples) · [Native experiment datasets](#/docs/guides/tracing)

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Choose a source and follow its native collection → preparation → training →
publication workflow. Each example is an independent uv project.

| Example | Source | Training SDK |
| --- | --- | --- |
| [HF agent tracing](#/docs/guides/example-hf-agent-tracing) | Hub OpenHands trajectories → native Pi sessions | Baseten Loops SFT + PPO |
| [Local TRL](#/docs/guides/example-trl) | Addition examples with Langfuse tracing | Local TRL SFT |
| [Harbor](#/docs/guides/example-harbor) | Harbor / Terminal-Bench jobs | Tinker / Spindle |
| [Harvey](#/docs/guides/example-harvey) | Harvey LAB transcripts and grades | Baseten Loops |
| [Inspect](#/docs/guides/example-inspect) | Inspect logs and Chartography images | Tinker / Spindle |
| [Ultrachat](#/docs/guides/example-ultrachat) | HF conversations | Fireworks |
| [Claude Agents SDK](#/docs/guides/example-claude-agents) | Langfuse experiments | Tinker |
| [Codex](#/docs/guides/example-codex) | Arize AX experiments | Tinker |
| [DeepAgents](#/docs/guides/example-deepagents) | LangSmith experiments | Spindle |
| [OpenAI Agents SDK](#/docs/guides/example-openai-agents) | Phoenix experiments | Tinker |
| [OpenCode](#/docs/guides/example-opencode) | LangSmith experiments | Tinker |

## Setup

Run commands from the selected example directory. Copy its `.env.example` to
`.env`, fill in provider credentials, and load them before collection. Use your
provider's supported model, tokenizer, training endpoint, and inference gateway.
Initialize task-data submodules with `git submodule update --init --recursive`.

## Policy updates

The RL examples use group-centered advantages and clipped PPO through native
SDKs. Keep failed attempts and collect at least two attempts per task. Flat
reward batches need more varied attempts before training.

Replay requires the collection checkpoint and optimizer state, complete actions,
the same tokenizer/renderer, and temperature 1 with untruncated sampling. Keep
publication receipts for subsequent rounds and evaluate on held-out tasks.
Ultrachat uses supervised fine-tuning.
