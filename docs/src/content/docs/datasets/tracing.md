Export OpenInference model calls from a native dataset experiment. Trains joins
each attempt to its dataset example, terminal grade, and linked trace.

## Before you start

Configure credentials and endpoints through the provider's SDK. Instrument model
calls with OpenInference and retain complete messages, tool schemas, and model IDs.
Dataset reference answers never become model-generated training targets.

| Provider | Trains extra | Native client | Experiment selection |
| --- | --- | --- | --- |
| Arize Phoenix | `arize-phoenix` | `phoenix.client.Client` / `AsyncClient` | Experiment ID |
| Arize AX | `arize-ax` | `arize.ArizeClient` | Experiment ID or name; optional space |
| LangSmith | `langsmith` | `langsmith.Client` | Experiment project name |
| Langfuse | `langfuse` | `langfuse.Langfuse` | Dataset name, experiment ID, start time |

Every built-in provider requires `experiment`. `dataset` validates the linked
dataset when supplied; Phoenix, AX, and LangSmith can resolve it from the
experiment. `score_name` selects a native evaluator metric. All graded attempts,
including zero rewards, keep their dataset-example group identity.

## Arize Phoenix

```python
from phoenix.client import Client as PhoenixClient
from trains.datasets import chat_dataset_from_tracing_provider
from trains.tracing import ArizePhoenixTracingProvider

provider = ArizePhoenixTracingProvider(
    PhoenixClient(), experiment="experiment-id", dataset="dataset-id", score_name="quality",
)
chats = chat_dataset_from_tracing_provider(provider)
chats.save_to_disk("var/chats/train")
```

Trains reads the experiment's pinned dataset version, task runs, evaluations, and
linked trace IDs. The experiment's native trace project is used by default.
Set `trace_project="my-agent"` when its runs link to traces stored in another project.

For an async client:

```python
from phoenix.client import AsyncClient
from trains.datasets import async_chat_dataset_from_tracing_provider

provider = ArizePhoenixTracingProvider(AsyncClient(), experiment="experiment-id")
chats = await async_chat_dataset_from_tracing_provider(provider)
```

## Arize AX

```python
from arize import ArizeClient
from trains.datasets import chat_dataset_from_tracing_provider
from trains.tracing import ArizeAXTracingProvider

provider = ArizeAXTracingProvider(
    ArizeClient(), experiment="experiment-id", dataset="dataset-id",
    space="space-id", score_name="quality",
)
chats = chat_dataset_from_tracing_provider(provider)
```

Trains follows native dataset versions, experiment runs, annotations, and linked
trace IDs across all pages. `space` resolves named resources. `trace_project`
overrides the native experiment trace project when the captured calls live elsewhere.

## LangSmith

```python
from langsmith import Client as LangSmithClient
from trains.datasets import chat_dataset_from_tracing_provider
from trains.tracing import LangSmithTracingProvider

provider = LangSmithTracingProvider(
    LangSmithClient(), experiment="experiment-project", dataset="dataset-name", score_name="quality",
)
chats = chat_dataset_from_tracing_provider(provider)
```

The experiment project supplies its dataset and recorded dataset version. Trains
exports its root attempts, follows their trace IDs, and joins native run feedback.
Use `trace_project` when model calls are stored in a separate tracing project.

## Langfuse

```python
from datetime import datetime, timezone
from langfuse import Langfuse
from trains.datasets import chat_dataset_from_tracing_provider
from trains.tracing import LangfuseTracingProvider

provider = LangfuseTracingProvider(
    Langfuse(), dataset="dataset-name", experiment="experiment-id",
    from_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc), score_name="quality",
)
chats = chat_dataset_from_tracing_provider(provider)
```

Langfuse's native Experiments API requires `from_start_time`; use a timezone-aware
time at or before the experiment starts. Trains selects only that experiment's dataset
items and follows their trace IDs, preserving expanded OpenInference metadata and
native item/trace scores.

## Formats and streaming

Pass a `TraceSpec` as the second constructor argument to select the model-call
format. `OpenInferenceTraceSpec(span_kind="AGENT")` selects descendant model calls;
exports retain parent spans for ancestry checks.

A replayable provider factory supports lazy HF datasets:

```python
from phoenix.client import Client as PhoenixClient
from trains.datasets import chat_dataset_from_tracing_provider
from trains.tracing import ArizePhoenixTracingProvider

def source():
    return ArizePhoenixTracingProvider(PhoenixClient(), experiment="experiment-id")

chats = chat_dataset_from_tracing_provider(source, streaming=True)
```

Keep the experiment and dataset version fixed across epochs. Missing trace
linkage, mismatched datasets, ambiguous scores, and duplicate linked traces fail
explicitly. Arbitrary query dictionaries and trace-only selection are unsupported.
For another source, implement the [provider protocol](#/docs/guides/sdk).
