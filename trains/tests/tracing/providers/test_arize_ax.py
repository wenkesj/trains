from types import SimpleNamespace
from unittest.mock import Mock

from openinference.semconv.trace import OpenInferenceSpanKindValues

from dataset_fixtures import span_frame
from trains.datasets import chat_dataset_from_tracing_provider
from trains.formats.tracing import OpenInferenceTraceSpec
from trains.tracing import ArizeAXTracingProvider


def test_native_export_mapping_grades_and_agent_descendants(processor, native_run):
    provider_type = ArizeAXTracingProvider
    export = Mock(return_value={"spans": span_frame().to_dict(orient="records"), "pagination": {}})
    client = SimpleNamespace(spans=SimpleNamespace(list=export))
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
    assert export.call_args.kwargs == {
        "project": "test",
        "filter": "context.trace_id = 'trace-native'",
    }
