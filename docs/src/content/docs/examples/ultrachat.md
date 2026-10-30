[Example source](https://github.com/wenkesj/trains/tree/main/examples/ultrachat) · [Datum conversion](#/docs/guides/datum)

> **Experimental:** Work in progress; convergence has not been demonstrated. Try a small run with a compatible local training service first. Hosted providers charge real money: estimate costs, verify credits, and stop unused resources.

Train complete conversations from the pinned `HuggingFaceH4/ultrachat_200k`
dataset using the native Fireworks SDK. Assistant tokens receive the loss weights.

## Train

From `examples/ultrachat`, configure `FIREWORKS_API_KEY` for your account:

```sh
uv sync --locked
uv run python run.py train \
  --base-model "$BASE_MODEL" --tokenizer-model "$TOKENIZER_MODEL" \
  --samples 32 --steps 1
```

The run uses one small batch and saves native state and sampler references to
`var/checkpoint.json`. Choose a trainable model and its matching tokenizer;
reduce the sample count when full conversations exceed its context budget.

## Publish

```sh
uv run python run.py deploy var/checkpoint.json \
  --account-id "$FIREWORKS_ACCOUNT_ID" \
  --model-id ultrachat-policy --deployment-id ultrachat-serving
```

The native SDK promotes the saved checkpoint, creates a deployment, and records
its model and callable URL in `var/deployment.json`. Use that receipt with the
Fireworks inference client. Saving sampler weights and creating a deployment
are separate operations.
