Render complete conversations into the native datum type expected by your training SDK.
Use the exact tokenizer or renderer used to collect the model calls.

## Convert conversations

```python
import json
from transformers import AutoTokenizer
from trains.datasets import chat_dataset_from_records, datum_dataset_from_chat_dataset
from trains.datum.adapters.tinker import TinkerDatumAdapter

chats = chat_dataset_from_records([{
    "messages": [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
    ],
}])
tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM3-Omni-2B", trust_remote_code=True)
datums = datum_dataset_from_chat_dataset(chats, tokenizer, TinkerDatumAdapter())
datums.save_to_disk("var/datums/train")
print(json.loads(datums[0]["datum_json"]))
```

Install `trains-sdk[tinker]` for this adapter. Baseten Loops, Fireworks, and Spindle
have their own adapters under `trains.datum.adapters`. Training uses the provider's
native client; Trains supplies datasets and datum conversion.

## Targets and masks

Each position predicts its next token. Assistant actions receive loss weights;
prompts, tool results, and prior context are masked. Tools and chat-template
settings stay with the conversation. Incompatible templates or incomplete model
outputs are rejected.

Rows contain `datum_json` alongside optional `trace_id`, `span_id`, `model_id`,
`group_id`, `reward`, and `metadata_json`. Native SDK types own their datum schemas.
`datum_dataset_from_datums` also accepts existing native datums directly.

## Media and rendering

Use a compatible processor or native Cookbook renderer for image/audio chunks.
Video rendering extracts timestamped frames. Resolve remote assets before saving
a portable chat dataset; SDK schema support and hosted model support are separate.

## Scale conversion

HF datasets support batching, multiprocessing, and caching. Supported streaming
builders use replayable generator factories. Preserve the collected checkpoint,
tokenizer, renderer, and complete model-visible history for policy replay.

The [Python reference](#/docs/reference/python/trains.datasets) describes builders
and adapter protocols. The [examples](#/docs/guides/examples) show native SDK use.
