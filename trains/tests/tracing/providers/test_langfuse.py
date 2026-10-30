from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

from langfuse.api.commons.types.observation_v2 import ObservationV2
from langfuse.api.observations.types.observations_v2response import ObservationsV2Response

from trains.datasets import chat_dataset_from_tracing_provider
from trains.formats.tracing import TraceSpec
from trains.tracing import LangfuseTracingProvider


def test_langfuse_cursor_keeps_generations_of_one_trace_together(processor, native_run):
    item = {
        "type": "GENERATION",
        "trace_id": "trace",
        "id": "span",
        "metadata": {
            "attributes": {
                "openinference.span.kind": "LLM",
                "llm.input_messages.0.message.role": "user",
                "llm.input_messages.0.message.content": "go",
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.content": "north",
            }
        },
    }
    get = Mock(
        side_effect=[
            {"data": [item], "meta": {"cursor": "next"}},
            {"data": [{**item, "id": "span-2"}], "meta": {}},
        ]
    )
    provider = LangfuseTracingProvider(
        SimpleNamespace(api=SimpleNamespace(observations=SimpleNamespace(get_many=get))),
        TraceSpec(),
        dataset="dataset",
        experiment="experiment",
        from_start_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    native_run.trace_id = "trace"
    provider.iter_experiment_runs = Mock(return_value=iter([native_run]))
    data = chat_dataset_from_tracing_provider(provider)
    assert len(data) == 2 and data[0]["trace_id"] == data[1]["trace_id"]
    assert get.call_args.kwargs["cursor"] == "next"


def test_langfuse_native_promoted_io_and_attribute_namespace_are_preserved(native_run):

    observation = ObservationV2(
        id="span",
        trace_id="trace",
        project_id="project",
        name="messages.create",
        start_time=datetime.now(UTC),
        type="GENERATION",
        input='{"system":"Be helpful.","messages":[{"role":"user","content":"Go north"}]}',
        output='{"role":"assistant","content":[{"type":"text","text":"Done."}]}',
        metadata={
            "attributes.openinference.span.kind": "LLM",
            "attributes.llm.system": "anthropic",
        },
    )
    get = Mock(return_value=ObservationsV2Response(data=[observation], meta={}))
    provider = LangfuseTracingProvider(
        SimpleNamespace(api=SimpleNamespace(observations=SimpleNamespace(get_many=get))),
        dataset="dataset",
        experiment="experiment",
        from_start_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    native_run.trace_id = "trace"
    provider.iter_experiment_runs = Mock(return_value=iter([native_run]))
    (trace,) = provider.iter_traces()
    (record,) = provider.spec.iter_chats(trace)
    assert record["messages"][0]["content"] == "Be helpful."
    assert record["messages"][-1]["content"] == "Done."
    assert get.call_count == 1  # expanding attributes covers its dotted descendants
