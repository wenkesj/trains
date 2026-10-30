# Trains Python library

Convert model traces and trajectories into Hugging Face datasets for native training SDKs.

Start with the [project README](../README.md#installation) for installation and supported workflows, or read the [user guide](https://wenkesj.github.io/trains/#/docs/guides/quickstart).

## Public API

| Import | Purpose |
| --- | --- |
| `trains.datasets` | Build native HF datasets from datums, traces, or ATIF files |
| `trains.sessions.adapters` | Convert native session files or in-memory sessions to chat records |
| `trains.datum.adapters` | Native datum Adapter protocol; optional SDK adapters |
| `trains.types` | Typed chat rows and provenance; native datums remain provider-defined |
| `trains.formats.records` | Typed HF/TRL records, messages, and tools |
| `trains.formats.tracing` | Select spans and convert model calls |
| `trains.formats.atif` | Read ATIF trajectories with optional native validation |
| `trains.tracing` | Connect initialized native tracing clients |

Dataset rows contain `datum_json`, `trace_id`, `span_id`, `model_id`, `group_id`, `reward`, and `metadata_json`. Save and load them with Hugging Face Datasets. See the [reference](https://wenkesj.github.io/trains/#/docs/guides/sdk) for signatures and column meanings.

Tracing instrumentation belongs to the native provider. Use OpenInference instrumentation with Arize AX, Phoenix, Langfuse, or
LangSmith. The library reads their span exports and native dataset experiments.

## Development

Follow the [workspace setup](../README.md#contributing), then run:

```bash
.venv/bin/ruff check trains
.venv/bin/pygate check trains
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest trains/tests -q
CUDA_VISIBLE_DEVICES="" .venv/bin/ty check --project trains --python .venv
```

Tests mirror package modules: `tests/datum/adapters/`, `tests/datasets/`, `tests/tracing/adapters/`, and `tests/tracing/providers/`. Shared fixtures live in `conftest.py` and support modules. Tests cover OpenInference exports and native experiment API contracts, provider pagination, tokenization, prompt masks, datum reconstruction, and HF persistence. Use small local tokenizers and sanitized fixtures for new coverage.

To add a provider or format, test its native export and complete model I/O,
including tools. Keep benchmark grading and training recipes in examples.
See the [API reference](https://wenkesj.github.io/trains/#/docs/guides/sdk) for adapter protocols and
[collection guide](https://wenkesj.github.io/trains/#/docs/guides/collection) for media support.
