# Examples

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Choose an input source, then follow its README to collect data, train, and use the
published policy. Each example is an independent uv project with a lockfile.
Run its commands from its own directory. Copy its `.env.example` to `.env`,
fill in credentials, and load it as shown in that README.

| Example | Input | Training |
| --- | --- | --- |
| [HF agent tracing](hf-agent-tracing/README.md) | Hub OpenHands trajectories → native Pi sessions | Baseten Loops SFT + PPO |
| [Local TRL](tracing/trl/README.md) | Addition examples with Langfuse observations | Local TRL SFT |
| [Harvey LAB](harvey_lab/README.md) | Native LAB artifacts | Baseten Loops |
| [UltraChat](ultrachat/README.md) | Hugging Face conversations | Fireworks |
| [Harbor](harbor/README.md) | Terminal-Bench / Harbor jobs | Tinker / Spindle |
| [Inspect](inspect/README.md) | Inspect logs, including Chartography vision tasks | Tinker / Spindle |
| [Tracing](tracing/README.md) | OpenAI Agents, Claude Agents, Codex, OpenCode, Deep Agents | Tinker / Spindle |

Harvey LAB and tracing tasks use pinned Git submodules. Initialize from the root:

```sh
git submodule update --init --recursive
```

Hosted training uses provider credentials and native SDKs. For local training,
configure [the trainer](../train/README.md), then start from the checkout root:

```sh
docker compose -f compose.train.yml --profile training --profile vlm up -d
```

Each README supplies its collection, training, publication, and inference commands.
Keep judges fixed when comparing the original and updated policy. Outputs stay in
`var/`; native Harvey LAB runs use its submodule's ignored `data/results/`.

## Reinforcement learning

The benchmark and tracing examples use [recipes.rl](recipes/README.md): group-centered
advantages and clipped PPO. UltraChat uses supervised fine-tuning.

## Checks

From the repository root, sync the tools and the example, then run its checks:

```sh
uv sync --locked --only-group quality
uv sync --project examples/harbor --locked --all-extras
.venv/bin/task example:lint example:type example:test EXAMPLE=examples/harbor
```

Replace `examples/harbor` with any example path. `task examples:quality` checks every
example with Ruff, Pygate, and ty after its environment is synced. CI discovers projects automatically and
runs their isolated checks and available CPU tests; it also runs example doctests.
