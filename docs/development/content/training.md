This guide trains from a Trains dataset with a native provider SDK. It covers supervised updates, policy scores, checkpoints, and sampling.

## Before you start

Prepare a dataset with the actual inference tokenizer using the [datasets tutorial](#/docs/guides/collection) or a [tracing provider](#/docs/guides/integrations). The examples below load `var/datums/train`.

You also need a training endpoint with the requested base model. The examples use
`http://127.0.0.1:8382`, the default address for a self-hosted training service.
Follow the [local service setup](https://github.com/wenkesj/trains/tree/main/train),
or use your hosted provider's endpoint, credentials, and supported model.

Choose one client below. The examples pin compatible native SDK versions and
document their compatibility overrides; Baseten training and management run
together in the same environment.

## Tinker and Modal Spindle

Install the native Tinker client:

```bash
uv add 'tinker==0.35.0'
```

```python
import tinker
from tinker import types

service = tinker.ServiceClient(
    base_url="http://127.0.0.1:8382",
    api_key="tml-local",
)
training = service.create_lora_training_client(
    base_model="LiquidAI/LFM3-Omni-2B", rank=8, train_unembed=False,
)
```

Modal Spindle uses the Tinker-compatible client contract. Use the endpoint supplied by your Spindle deployment when connecting to that service.

## Baseten Loops

In your Baseten project's environment:

```bash
uv add 'baseten-loops==0.32.0'
```

```python
from baseten import loops

types = loops
service = loops.ServiceClient(
    base_url="http://127.0.0.1:8382", api_key="local",
)
training = service.create_lora_training_client(
    base_model="LiquidAI/LFM3-Omni-2B", rank=8,
)
```

Use the factory arguments supported by this client. In particular, this factory accepts the rank while Tinker's additional `train_unembed` argument belongs to its own client interface.

## Fireworks Firetitan

In your Fireworks project's environment:

```bash
uv add 'fireworks-ai[training-sdk]==1.2.27'
```

```python
from fireworks.training import sdk

types = sdk
service = sdk.ServiceClient(
    base_url="http://127.0.0.1:8382", api_key="tml-local",
)
training = service.create_lora_training_client(
    base_model="LiquidAI/LFM3-Omni-2B", rank=8, train_unembed=False,
)
```

The local service accepts the dummy credentials shown here and implements the supported text-LoRA contract. Cloud provisioning and deployment use the hosted provider's APIs.

## Load native datums

Continue with the `training` client and `types` module from your chosen provider:

```python
import json
from datasets import load_from_disk

hf_dataset = load_from_disk("var/datums/train")

def parse_datum(value):
    value = json.loads(value)
    return types.Datum(
        model_input=types.ModelInput.model_validate(value["model_input"]),
        loss_fn_inputs={
            name: types.TensorData(**tensor)
            for name, tensor in value["loss_fn_inputs"].items()
        },
    )
```

These constructors read the text datums produced by the built-in adapters and support the pinned native clients, including Tinker's dataclass-based `Datum`. Trains's JSON column preserves the native token and loss-input layout.

## Train with cross-entropy

This supervised example trains on the selected outputs for three epochs. Trains-generated datums include prompt masks in `weights`:

```python
for epoch in range(3):
    shuffled = hf_dataset.shuffle(seed=42 + epoch)
    for batch in shuffled.iter(batch_size=32):
        datums = [parse_datum(value) for value in batch["datum_json"]]
        count = sum(
            sum(weight != 0 for weight in datum.loss_fn_inputs["weights"].data)
            for datum in datums
        )
        if count == 0:
            continue
        for datum in datums:
            weights = datum.loss_fn_inputs["weights"].data
            datum.loss_fn_inputs["weights"] = types.TensorData(
                data=[weight / count for weight in weights],
                dtype="float32",
                shape=[len(weights)],
            )
        training.forward_backward(datums, loss_fn="cross_entropy").result()
        training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
```

Weights are normalized over active target tokens in each optimizer batch. For gradient accumulation, keep the normalization consistent across all batches contributing to the same optimizer step. Raw native datums should supply the masking and reduction required by your objective.

To train only on successful attempts, filter the HF dataset by your evaluator's reward before iterating. This is reward-filtered supervised imitation.

## Use the Tinker cookbook

With the Tinker client selected, reuse `parse_datum` in the cookbook's native HF adapter:

```python
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset

supervised = SupervisedDatasetFromHFDataset(
    hf_dataset=hf_dataset,
    batch_size=32,
    map_fn=lambda row: parse_datum(row["datum_json"]),
)
```

Install the cookbook in your Tinker project. Its [supervised dataset adapter](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/supervised/data.py) counts complete batches. Use HF `.iter(batch_size=...)` when retaining a partial final batch matters.

## Prepare a policy objective

Harbor, Harvey LAB, and Inspect use group mean-centered advantages and clipped PPO,
with no reward-filtered supervised fallback. UltraChat remains supervised.
Collect at least two independently graded rollouts per task and retain failures.
The baseline counts unique rollouts, not calls or rendered chunks. A batch with
no within-group reward variation stops before updating.

The separate `examples/recipes` project provides the shared `recipes.rl` recipe; it
is not part of Trains’s API. Install it in your example environment
(`uv add --editable ../recipes` from a sibling example directory).
Use the example helper after restoring the exact collection checkpoint with its
optimizer. Each rendered row must retain `group_id`, `trace_id`, `reward`,
`model_id`, `metadata_json.sampling`, and its native `datum_json`:

```python
from recipes.rl import ppo_update

training = service.create_training_client_from_state_with_optimizer(behavior_state)
metrics = ppo_update(
    list(hf_dataset), training, types,
    behavior_model=collected_model,
    steps=1,
    learning_rate=1e-5,
)
```

For Fireworks, restore a matching native client with
`training.load_state_with_optimizer(behavior_state).result()` instead.
The helper scores old-policy logprobs once with a forward-only operation,
freezes them, and sends PPO exactly `target_tokens`, `logprobs`, and
`advantages`. It masks context with zero advantages and clips ratios at
`[0.8, 1.2]`. It does not send CE weights to PPO or perform a supervised backward
pass. Advantages are reward minus group mean, without standard-deviation
normalization, scaled by total action weight for a token-mean objective.

These are teacher-forced old-policy logprobs, not captured sampling logprobs.
Replay requires the identical policy, tokenizer and renderer, complete model-visible
actions including reasoning, temperature 1 and untruncated sampling. Importers
retain model identity and available sampling settings; explicit mismatches are
rejected. External serving defaults and train/serve equivalence still require
configuration. No train/serve drift correction or MoE router replay is provided.
Harvey's stock artifacts omit hidden reasoning, so its example requires a
non-reasoning policy. See the [example contract and provider sources](https://github.com/wenkesj/trains/tree/main/examples#reinforcement-learning)
for collection commands, deployment receipts and replay requirements.

Teacher scoring, KL objectives, and distillation require their own native training recipes. The local service's supported objectives are listed in its [README](https://github.com/wenkesj/trains/tree/main/train).

## Save and resume

```python
state = training.save_state("iteration-1-state").result()
sampler = training.save_weights_for_sampler("iteration-1-policy").result()
print(state.path, sampler.path)
```

Persist the returned native URIs with the collection and model identity. The state checkpoint retains optimizer state; the sampler checkpoint provides inference weights.

Tinker/Spindle and Baseten support `service.create_training_client_from_state_with_optimizer(state.path)`. With Fireworks, create a matching LoRA client and call `training.load_state_with_optimizer(state.path).result()`. Preserve the original model, rank, and alpha configuration.

Save completed operation receipts in your application. After an uncertain outcome, inspect the operation or resume from a known checkpoint before submitting further updates.

## Sample or deploy

Tinker/Spindle and Baseten can create a native sampling client with `service.create_sampling_client(model_path=sampler.path)`.

Fireworks supports an explicit deployment sampler:

```python
from fireworks.training.sdk.sampling import DeploymentSampler

sampling = service.create_sampling_client(
    deployment_sampler=DeploymentSampler(
        inference_url="http://127.0.0.1:8382",
        model=sampler.path,
        api_key="local",
    )
)
```

Use the hosted deployment's inference URL for a cloud sampler.

For local application inference, configure the training service's `TRAIN_INFERENCE_*` settings using the [publication setup](https://github.com/wenkesj/trains/tree/main/train#publish-and-serve). The service owns adapter paths, loading, and shared-GPU sleep/wake. With Tinker/Spindle, the native `save_weights_for_sampler()` call above already publishes the model: use its returned URI directly with an OpenAI-compatible inference client. No separate deployment request is needed:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8382/oai/api/v1",
    api_key="tml-local",
)
response = client.chat.completions.create(
    model=sampler.path,
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message)
```

The Harbor and Inspect examples use this native Tinker pathway. They return the sampler URI for the next collection, and the training service loads the adapter into vLLM on the first inference request. Keep the optimizer-state URI separately for training resumption.

The local service also implements Fireworks checkpoint promotion followed by deployment or LoRA attachment, and Baseten checkpoint deployment followed by deployment-scoped chat inference. Both providers' native management SDKs accept local base URLs; the [provider pathway examples](https://github.com/wenkesj/trains/tree/main/train#provider-deployment-pathways) show the tested clients and IDs. An optional `POST /v1/deployments` convenience API creates named application aliases. These pathways emulate publication on preconfigured local hardware; hosted GPU provisioning remains provider-specific.

## Close the client

Use the shutdown method for the selected SDK after outstanding work has completed:

| Client | Shutdown |
| --- | --- |
| Tinker/Spindle | `service.close("success").result()` |
| Baseten Loops | `training.close()` |
| Fireworks | `service.close()` |

A transport's shutdown and a hosted trainer's provisioning lifecycle may be separate operations. Follow the provider's lifecycle API for hosted resources.
