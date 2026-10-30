# Local training service

Run native Tinker, Spindle, Baseten Loops, and Fireworks training code against
local Hugging Face models. The service trains PEFT LoRA adapters, saves optimizer
state, and serves published checkpoints through SGLang or vLLM.

## Start a server

Requires the [workspace environment](../README.md#contributing) and a complete
local model snapshot with tokenizer/processor files. From the checkout root:

```sh
TRAIN_TRUST_REMOTE_CODE=true uv run train serve \
  --model LiquidAI/LFM3-Omni-2B=/absolute/path/to/model/snapshot --root var/train
```

The API listens on `127.0.0.1:8382`. For CUDA training and SGLang, set
`TRAIN_MODEL_PATH=./var/models/lfm3-omni` in `.env`, download the pinned model,
and start the services:

```sh
uvx --from huggingface-hub hf download LiquidAI/LFM3-Omni-2B \
  --revision 5797d07c125503bd4f4c0e7f915ae2d3d703e8de --local-dir var/models/lfm3-omni
docker compose -f compose.train.yml -f compose.train.lora.yml --profile training --profile vlm up -d
```

The pinned SGLang image includes LFM3 support under development upstream.
Compose enables shared-GPU coordination and untruncated sampling defaults. Route inference
through the training API so its lock coordinates training, loading, and streaming.
Host `train serve` processes do not load `.env` automatically.

## Train with Tinker

```python
import tinker
from tinker import types

service = tinker.ServiceClient(
    base_url="http://127.0.0.1:8382",
    api_key="tml-local",
)
training = service.create_lora_training_client(
    base_model="LiquidAI/LFM3-Omni-2B",
    rank=8,
    train_unembed=False,
)
tokenizer = training.get_tokenizer()
tokens = tokenizer.encode("2 + 2 = 4", add_special_tokens=True)
datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokens[:-1]),
    loss_fn_inputs={
        "target_tokens": types.TensorData(
            data=tokens[1:],
            dtype="int64",
            shape=[len(tokens) - 1],
        ),
    },
)
training.forward_backward([datum], "cross_entropy").result()
training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
state = training.save_state("step-1").result()
print(state.path)
service.close("success").result()
```

The dummy key's `tml-` prefix satisfies Tinker's client validation. Hosted
providers require their own credentials. Follow the [examples](../examples/README.md)
for dataset preparation, PPO, publication, and inference with each native SDK.

## Supported clients

| Client | Tested contract | Routes |
| --- | --- | --- |
| Tinker | `tinker==0.35.0` | `/api/v1` futures and tensors |
| Modal Spindle | Tinker-compatible client contract | `/api/v1` |
| Baseten Loops | `baseten-loops==0.32.0` | `/v1/loops` and operation polling |
| Fireworks Firetitan | `fireworks-ai[training-sdk]==1.2.27` | `/api/v1` and token completions |

Supported operations include LoRA creation, forward/backward, AdamW,
cross-entropy, importance sampling, PPO, state save/restore, and native sampling.
Images and extracted video frames use compatible HF vision models and processors.
DMel audio is accepted at the wire boundary but rejected by the local backend.
Resolve image asset pointers to bytes before submission.

Cloud provisioning, billing, distributed training, and full-parameter training
are outside this service's contract. Hosted modality support depends on the
provider's deployed model.

## Publish and serve

Native sampler publication produces immutable checkpoint references. Keep those
receipts; clients need no checkpoint filesystem access. The recommended Compose
setup serves Liquid Omni adapters directly through SGLang:

```sh
export TRAIN_INFERENCE_BACKEND=sglang
export TRAIN_INFERENCE_CHECKPOINT_MODE=lora
export TRAIN_INFERENCE_URL=http://127.0.0.1:30000
export TRAIN_INFERENCE_TRAIN_ROOT=/train
export TRAIN_INFERENCE_BASE_MODEL=LiquidAI/LFM3-Omni-2B
```

`TRAIN_INFERENCE_TRAIN_ROOT` is the training root as mounted inside the inference server. Compose sets
these paths already. Use `TRAIN_INFERENCE_API_KEY` for backend authentication and
`TRAIN_INFERENCE_SERVED_MODEL` when its served name differs from the base model.

Set `TRAIN_INFERENCE_MAX_OUTPUT_TOKENS=4096` to cap generated tokens for clients with
larger default budgets, such as Harvey's judge. SGLang text chat also uses native
tokenization to fit the output budget into the remaining context. Inputs remain
intact; overflowing inputs are rejected. Without this setting the API preserves
the client's output budget. Compose sets a 4096-token ceiling.

| Native publication | Local inference route |
| --- | --- |
| Tinker / Spindle: save sampler weights; use the sampler URI as model | `/oai/api/v1` |
| Fireworks: list/promote checkpoint, create deployment | `/inference/v1` |
| Baseten: list checkpoint IDs, deploy through management SDK | `/deployment/{id}/sync/v1` |

The generic `/v1` gateway supports Chat Completions, Responses, and Anthropic
messages through the configured backend. Tools, streaming, and inference
capabilities follow that backend. The [API reference](../docs/README.md#api-reference)
also documents explicit `/v1/deployments` publication and deletion.

The LoRA override patches the pinned SGLang image for Liquid's language layers.
Rank 8 and decode CUDA graphs are the tested defaults. Set
`INFERENCE_DISABLE_CUDA_GRAPH=true` to disable graphs. HF and SGLang token scores
still differ; evaluate quality on your task. Save active trainer state before
switching configurations.

Omit `-f compose.train.lora.yml` for merged SGLang reloads. BF16 merging can round
away small adapter updates; native LoRA avoids that weight-rounding step.

For an external vLLM server, use `TRAIN_INFERENCE_BACKEND=vllm` and
`TRAIN_INFERENCE_CHECKPOINT_MODE=lora` with runtime LoRA loading enabled.

Published bindings survive service restarts and reload checkpoints on demand.
Checkpoint aliases remain pinned to immutable snapshots. Sessions and pending
futures belong to the process: save state before restarting, then restore it in
a new client. Preserve caches, checkpoints, and persisted volumes.

## Fixed LAB judge

Create a separate immutable Liquid judge with the Harvey example CLI:

```sh
cd examples/harvey_lab
export BASETEN_API_KEY=local
uv run python run.py initialize --training-url http://127.0.0.1:8382 \
  --sampler-name gpt-4-lfm3-fixed-judge --output var/judge-checkpoint.json
uv run python run.py deploy var/judge-checkpoint.json --local \
  --control-url http://127.0.0.1:8382 --model-name harvey-fixed-judge \
  --instance-type local --hf-secret unused-local --output var/judge.json
```

Set `OPENAI_BASE_URL` and LAB's `--judges` from that receipt's `inference_url` and
`model`; use a dummy local `OPENAI_API_KEY`. The `gpt-4-` prefix selects LAB's
OpenAI protocol while the weights remain Liquid. Keep this deployment fixed
across training rounds.

## GPU configuration

The Docker profile supplies CUDA 13, PyTorch, and pinned kernels. Initial use
compiles kernels into a persistent cache. Monitor host and GPU memory.
Enable gradient checkpointing for longer contexts that exceed available GPU memory.
Native token sampling accepts output budgets that fit the registered model context
and stops at the model’s end-of-sequence token or the supplied stop sequences.
The optional SGLang sampler keeps token IDs and log probabilities from the backend
and shares the GPU lock with training. Media, string stops, other temperatures,
and targeted prompt scoring use the Transformers sampler.

| Setting | Purpose |
| --- | --- |
| `TRAIN_DEVICE` | Select the training device, for example `cuda:0` |
| `TRAIN_DTYPE` | Select `float32` or `bfloat16` |
| `TRAIN_GRADIENT_CHECKPOINTING` | Recompute activations during backward |
| `TRAIN_CHECKPOINT_OFFLOAD` | Retain checkpoint inputs in host memory |
| `TRAIN_MLP_CHUNK_SIZE` | Bound feed-forward activation chunks; Compose defaults to 2048 tokens |
| `TRAIN_SAMPLING_BACKEND` | `sglang` uses native token generation for greedy or untruncated temperature-one text requests; default `transformers` |
| `TRAIN_ATTENTION_BACKEND` | `auto` selects installed FlashAttention 4 for CUDA BF16; `sdpa` forces PyTorch attention |
| `TRAIN_KERNELS` | Select automatic kernels or the `reference` implementation |
| `TRAIN_MEMORY_LIMIT` / `TRAIN_MEMORY_SWAP_LIMIT` | Override Compose host-memory limits |

## Development

From the checkout root:

```sh
.venv/bin/ruff check train
.venv/bin/pygate check train
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest train/tests -q
CUDA_VISIBLE_DEVICES="" .venv/bin/ty check --project train --python .venv
```

The default suite uses tiny CPU models. Set `TRAIN_TEST_VENDOR_SDK=1` for native
Baseten/Fireworks contracts and `TRAIN_TEST_CUDA=1` for an available GPU.
Test actual native clients when changing compatibility. Regenerate the
[OpenAPI snapshot](../docs/README.md#api-reference) after API changes.
