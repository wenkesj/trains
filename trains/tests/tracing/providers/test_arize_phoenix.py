import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openinference.semconv.trace import OpenInferenceSpanKindValues
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from dataset_fixtures import span_frame
from trains.datasets import (
    async_chat_dataset_from_tracing_provider,
    chat_dataset_from_tracing_provider,
)
from trains.formats.tracing import OpenInferenceTraceSpec, TraceSpec
from trains.tracing import ArizePhoenixTracingProvider


def test_phoenix_async_client_matches_sync_dataset(processor, phoenix_async_client):
    provider = ArizePhoenixTracingProvider(
        phoenix_async_client, OpenInferenceTraceSpec(), experiment="experiment"
    )
    data = asyncio.run(async_chat_dataset_from_tracing_provider(provider))
    assert len(data) == 1 and data[0]["span_id"] == "llm"


def test_native_export_mapping_grades_and_agent_descendants(processor, native_run):
    provider_type = ArizePhoenixTracingProvider
    export = Mock(return_value=span_frame())
    client = SimpleNamespace(export_to_df=export, spans=SimpleNamespace(get_spans_dataframe=export))
    spec = OpenInferenceTraceSpec.from_span_kind(OpenInferenceSpanKindValues.AGENT)
    provider = provider_type(
        client,
        spec,
        experiment="experiment",
        trace_project="test",
        score_name="quality",
    )
    provider.iter_experiment_runs = Mock(return_value=iter([native_run]))
    data = chat_dataset_from_tracing_provider(provider)
    assert len(data) == 1 and data[0]["trace_id"] == "trace-native"
    assert data[0]["reward"] == 0.75 and data[0]["group_id"] == "task-1"
    assert data[0]["span_id"] == "llm"
    assert export.call_args.kwargs["project_identifier"] == "test"
    assert "trace-native" in str(export.call_args.kwargs["query"])


import os
import time
from uuid import uuid4

import httpx
from phoenix.client import Client
from phoenix.client.types.spans import SpanQuery
from phoenix.otel import register


@pytest.mark.skipif(not os.environ.get("TRAINS_TEST_PHOENIX_URL"), reason="requires native Phoenix")
def test_native_phoenix_export_grade_and_streaming_repeat():
    url = os.environ["TRAINS_TEST_PHOENIX_URL"]
    project = "trains-contract-" + uuid4().hex
    provider = register(
        project_name=project,
        endpoint=url.rstrip("/") + "/v1/traces",
        protocol="http/protobuf",
        batch=False,
        auto_instrument=False,
        set_global_tracer_provider=False,
        verbose=False,
    )
    payload = {
        "request": {"messages": [{"role": "user", "content": "go"}]},
        "response": {"choices": [{"message": {"role": "assistant", "content": "north"}}]},
    }
    with provider.get_tracer("contract").start_as_current_span("model-call") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("input.value", json.dumps(payload["request"]))
        span.set_attribute("output.value", json.dumps(payload["response"]))
        trace_id = f"{span.get_span_context().trace_id:032x}"
        span_id = f"{span.get_span_context().span_id:016x}"
    assert provider.force_flush()
    provider.shutdown()
    raw = Tokenizer(
        WordLevel({"[UNK]": 0, "user": 1, "go": 2, "assistant": 3, "north": 4}, unk_token="[UNK]")
    )
    raw.pre_tokenizer = Whitespace()
    processor = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    processor.chat_template = "{% for m in messages %}{{m.role + ' ' + m.content + ' '}}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    client = Client(base_url=url)
    query = {
        "project_identifier": project,
        "query": SpanQuery().where(f"context.trace_id == '{trace_id}'"),
    }
    deadline = time.monotonic() + 10
    while True:
        try:
            spans = client.spans.get_spans_dataframe(**query)
            if spans is not None and not spans.empty:
                break
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
        if time.monotonic() > deadline:
            raise AssertionError("Phoenix did not finish ingesting the exported span")
        time.sleep(0.05)
    client.spans.add_span_annotation(
        span_id=span_id, annotation_name="reward", annotator_kind="CODE", score=0.75, sync=True
    )
    native_dataset = client.datasets.create_dataset(
        name="contract-" + uuid4().hex, inputs=[{"task": "go"}], outputs=[{}]
    )
    experiment = client.experiments.create(
        dataset_id=native_dataset.id, dataset_version_id=native_dataset.version_id
    )
    run = client.experiments.log_run(
        experiment_id=experiment["id"],
        dataset_example_id=native_dataset.examples[0]["node_id"],
        repetition_number=1,
        output=payload["response"],
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
        trace_id=trace_id,
    )
    client.experiments.log_evaluation(experiment_run_id=run["id"], name="quality", score=0.75)
    source = ArizePhoenixTracingProvider(
        client,
        TraceSpec(),
        experiment=experiment["id"],
        dataset=native_dataset.id,
        trace_project=project,
        score_name="quality",
    )
    dataset = chat_dataset_from_tracing_provider(source)
    assert len(dataset) == 1 and dataset[0]["trace_id"] == trace_id
    assert dataset[0]["reward"] == 0.75

    def source_factory():
        return ArizePhoenixTracingProvider(
            Client(base_url=url),
            TraceSpec(),
            experiment=experiment["id"],
            dataset=native_dataset.id,
            trace_project=project,
            score_name="quality",
        )

    stream = chat_dataset_from_tracing_provider(source_factory, streaming=True)
    assert list(stream) == list(stream) == dataset.to_list()


def test_http_telemetry_can_share_the_openinference_trace(processor, native_run):
    frame = span_frame()
    frame.loc[len(frame)] = {
        "context.trace_id": "trace-native",
        "context.span_id": "http",
        "parent_id": "root",
        "attributes.openinference.span.kind": "",
        "attributes.input.value": "",
        "attributes.output.value": "",
    }
    client = SimpleNamespace(spans=SimpleNamespace(get_spans_dataframe=Mock(return_value=frame)))
    provider = ArizePhoenixTracingProvider(
        client, OpenInferenceTraceSpec(), experiment="experiment"
    )
    provider.iter_experiment_runs = Mock(return_value=iter([native_run]))
    (trace,) = provider.iter_traces()
    assert len(trace.spans) == 3
    (record,) = provider.spec.iter_chats(trace)
    assert record["span_id"] == "llm"
