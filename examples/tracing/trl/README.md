# Local TRL training with Langfuse

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

[Example source](https://github.com/wenkesj/trains/tree/main/examples/tracing/trl) · [Native experiment datasets](https://wenkesj.github.io/trains/#/docs/guides/tracing)

Fine-tune `LiquidAI/LFM3-Omni-2B` with TRL and language-only LoRA. Vision and
audio weights stay frozen. Training runs in this process on a CUDA GPU;
no hosted training service is required.

Conversations pass through Trains' `chat_dataset_from_records` and `encode` APIs
before becoming TRL inputs. Trains validates messages and computes assistant-only
loss masks using the model's native chat template. TRL consumes those masks for
both the training and evaluation datasets.

```sh
cd examples/tracing/trl
uv sync
cp .env.example .env
# Configure Langfuse, then load it:
set -a; source .env; set +a
uv run python run.py --steps 200
```

The example trains on 256 four-digit addition problems and evaluates 64 disjoint
problems before training and every 100 steps. Langfuse records actual step
metrics and greedy generations with exact-match scores. `results.json` retains
the fixed split, answers, scores, and training history; `adapter/` contains the
LoRA weights and tokenizer under `artifacts/addition`.

Use a fresh `--output` directory for each run. `--steps`, `--eval-every`,
`--train-examples`, and `--eval-examples` adjust the schedule. Set
`LANGFUSE_TRACING_ENABLED=false` to train without exporting traces. Accuracy
changes must be assessed from the saved evaluations; completion alone does not
establish improvement.
