# Shared recipes

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

This independent package holds recipes shared by the benchmark and tracing
examples. `recipes.rl` supplies PPO helpers; each consuming project supplies its
native provider client.

```python
from recipes.rl import group_advantages, ppo_update, validate_sampling
```

Advantages are `reward - mean(task rewards)`, computed once per rollout, including
failures. All calls and chunks from a rollout share its advantage. Groups need
at least two rollouts; an entirely flat batch fails before an update.

PPO clips ratios to `[0.8, 1.2]`, without a critic or reference-policy KL term.
Only assistant action tokens receive advantages, normalized by total action
weight. The native loss receives target tokens, logprobs, and advantages.

From this directory, run `uv sync --locked` and `uv run pytest -q`.
