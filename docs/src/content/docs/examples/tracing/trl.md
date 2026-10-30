[Example source](https://github.com/wenkesj/trains/tree/main/examples/tracing/trl) · [Native experiment datasets](#/docs/guides/tracing)

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Fine-tune `LiquidAI/LFM3-Omni-2B` directly with TRL and language-only LoRA on a
CUDA GPU. Langfuse records training metrics and greedy generations with exact-match
scores. The Omni vision and audio weights stay frozen.

Conversations pass through Trains' `chat_dataset_from_records` and `encode` APIs
before becoming TRL inputs. Trains validates messages and computes assistant-only
loss masks using the model's native chat template. TRL consumes those masks for
both the training and evaluation datasets.

```sh
cd examples/tracing/trl
uv sync --python 3.12
# Configure Langfuse using .env.example.
uv run python run.py --steps 200
```

Training uses 256 four-digit addition problems and 64 disjoint evaluation
problems. The same held-out set is scored before training and every 100 steps.
`artifacts/addition/results.json` retains the split, answers, metrics, and scores;
`adapter/` contains the LoRA weights and tokenizer. Use a fresh `--output`
directory for each run. Set `LANGFUSE_TRACING_ENABLED=false` to disable export.
