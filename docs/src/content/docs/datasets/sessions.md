Native agent session artifacts can become HF chat records without exporting from
a tracing service. Session adapters preserve text, reasoning where available,
tool calls, tool results, and session identity. Dataset validation checks tool IDs
and conversation boundaries before training.

## Local files

```python
from trains.datasets import chat_dataset_from_sessions
from trains.sessions.adapters import CodexSessionAdapter

chats = chat_dataset_from_sessions(
    "~/.codex/sessions", adapter=CodexSessionAdapter()
)
chats.save_to_disk("var/codex-chats")
```

The path can identify one artifact or a directory searched recursively. Each
file becomes one conversation. Invalid artifacts raise an error with their path;
failed imports are never silently discarded.

| Adapter | Input |
| --- | --- |
| `CodexSessionAdapter` | Codex rollout JSONL, including `~/.codex/archived_sessions` |
| `OpenCodeSessionAdapter` | JSON from `opencode export` (v1) or `opencode session export` (v2) |
| `PiSessionAdapter` | Pi session JSONL, normally under `~/.pi/agent/sessions` |
| `STSSessionAdapter` | Hugging Face Session Trace Simple Format JSONL |

Codex imports `response_item` records rather than duplicated UI events. OpenCode
requires an export rather than its live database. Pi's unsupported content blocks
and OpenCode file attachments raise errors so assets can be handled explicitly.

## Sessions already in a Hugging Face dataset

When each row contains an entire raw session, map the adapter directly:

```python
from datasets import Features, Json, Value
from trains.sessions.adapters import PiSessionAdapter

adapter = PiSessionAdapter()
chat_dataset = dataset.map(
    lambda row: adapter.to_chat(row["session"]),
    remove_columns=dataset.column_names,
    features=Features({"messages": Json(), "trace_id": Value("string")}),
)
```

`to_chat` accepts a raw JSONL string or a list of decoded event objects. OpenCode's
method accepts its exported JSON string or decoded object. The explicit JSON
feature preserves heterogeneous tool arguments without Arrow schema inference.
Each row must contain a whole session; an individual timeline event is not a
conversation. For raw JSONL files on the Hub, load them as text documents with
`load_dataset("text", data_files=files, sample_by="document")` before mapping.

The [HF agent tracing example](#/docs/guides/example-hf-agent-tracing)
shows this mapping on pinned Hub sessions, Baseten Loops SFT, native Pi collection,
and PPO from local sessions. See the [Hub agent traces guide](https://huggingface.co/docs/hub/en/agent-traces)
for the source formats and local storage locations.

## Custom adapters

Implement `SessionAdapter[SessionT]` with `to_chat(session)` returning a chat
record, `read(path)` delegating file decoding to that conversion, and a `pattern`
such as `"*.jsonl"` for directory discovery. `SessionT` describes the adapter's
in-memory input type. The dataset loader accepts this protocol directly.

## Runnable examples

| Example | Workflow |
| --- | --- |
| [HF agent tracing](#/docs/guides/example-hf-agent-tracing) | Map Hub sessions to chats, train with Baseten Loops, collect native Pi sessions, and apply PPO |
| [Local TRL and Langfuse](#/docs/guides/example-trl) | Train a small Liquid model locally and record training metrics and generations |

See [all examples](#/docs/guides/examples) for benchmark and tracing-provider workflows.
