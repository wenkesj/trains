import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, create_autospec

import pytest
from arize._generated.api_client.models.experiment_run import ExperimentRun
from arize.datasets.client import DatasetsClient as AXDatasets
from arize.experiments.client import ExperimentsClient as AXExperiments
from arize.spans.client import SpansClient
from langfuse.api.experiments.client import ExperimentsClient as LangfuseExperiments
from langfuse.api.experiments.types.experiment_item import ExperimentItem
from langfuse.api.experiments.types.experiment_items_response import ExperimentItemsResponse
from langfuse.api.observations.client import ObservationsClient
from langfuse.api.scores_v3.client import ScoresV3Client
from langsmith import Client as LangSmithClient
from phoenix.client.resources.datasets import Dataset as PhoenixDataset
from phoenix.client.resources.datasets import Datasets as PhoenixDatasets
from phoenix.client.resources.experiments import Experiments as PhoenixExperiments

from dataset_fixtures import span_frame
from trains.datasets import chat_dataset_from_tracing_provider
from trains.formats.tracing import TraceRecord
from trains.tracing import (
    ArizeAXTracingProvider,
    ArizePhoenixTracingProvider,
    LangfuseTracingProvider,
    LangSmithTracingProvider,
)
from trains.tracing.providers.experiments import ExperimentRun as NativeExperimentRun
from trains.tracing.providers.experiments import _selected_score
from trains.tracing.providers.native import cursor_items, native_mapping


@pytest.mark.parametrize("attempt_error", [None, "Agent turn limit reached"])
def test_phoenix_experiment_joins_versioned_examples_repetitions_and_evaluations(attempt_error):
    experiments = create_autospec(PhoenixExperiments, instance=True)
    datasets = create_autospec(PhoenixDatasets, instance=True)
    runs = [
        {
            "id": f"run-{i}",
            "dataset_example_id": "example-node",
            "trace_id": f"trace-{i}",
            "error": attempt_error,
        }
        for i in range(2)
    ]
    experiments.get_experiment.return_value = {
        "experiment_id": "experiment",
        "dataset_id": "dataset",
        "dataset_version_id": "version",
        "project_name": "project",
        "task_runs": runs,
        "evaluation_runs": [
            {"experiment_run_id": run["id"], "name": "quality", "result": {"score": i / 2}}
            for i, run in enumerate(runs)
        ],
    }
    datasets.get_dataset.return_value = PhoenixDataset(
        {"id": "dataset", "name": "dataset"},
        {
            "version_id": "version",
            "examples": [
                {
                    "id": "example",
                    "node_id": "example-node",
                    "output": {"answer": "reference, never a model target"},
                }
            ],
        },
    )

    def fetch(**query):
        frame = span_frame()
        frame["context.trace_id"] = f"trace-{fetch.calls}"
        fetch.calls += 1
        return frame

    fetch.calls = 0
    client = SimpleNamespace(
        experiments=experiments,
        datasets=datasets,
        spans=SimpleNamespace(get_spans_dataframe=Mock(side_effect=fetch)),
    )
    provider = ArizePhoenixTracingProvider(client, experiment="experiment", score_name="quality")
    records = list(provider.iter_traces())
    assert [r.metadata["reward"] for r in records] == [0, 0.5]
    assert [r.metadata["example_id"] for r in records] == ["example", "example"]
    assert records[1].metadata["run_id"] == "run-1"
    assert next(provider.spec.iter_chats(records[0]))["group_id"] == "example"
    datasets.get_dataset.assert_called_once_with(dataset="dataset", version_id="version")
    assert "trace-0" in str(client.spans.get_spans_dataframe.call_args_list[0])


@pytest.mark.parametrize("attempt_error", [None, "Agent turn limit reached"])
def test_ax_native_experiment_and_span_clients_follow_all_pages(attempt_error):
    experiments = create_autospec(AXExperiments, instance=True)
    datasets = create_autospec(AXDatasets, instance=True)
    spans = create_autospec(SpansClient, instance=True)
    experiments.get.return_value = {
        "id": "experiment",
        "dataset_id": "dataset",
        "dataset_version_id": "version",
        "experiment_traces_project_id": "project",
    }
    datasets.get.return_value = {"id": "dataset"}
    datasets.list_examples.side_effect = [
        {
            "examples": [{"id": "other"}],
            "pagination": {"has_more": True, "next_cursor": "examples-next"},
        },
        {"examples": [{"id": "example"}], "pagination": {"has_more": False}},
    ]
    experiments.list_runs.return_value = {
        "experiment_runs": [
            {
                "id": "run",
                "example_id": "example",
                "result.trace.id": "trace-native",
                "error": attempt_error,
                "eval.quality.score": 0 if attempt_error else 0.75,
            }
        ],
        "pagination": {},
    }
    rows = span_frame().to_dict(orient="records")
    spans.list.side_effect = [
        {"spans": rows[:1], "pagination": {"next_cursor": "spans-next", "has_more": True}},
        {"spans": rows[1:], "pagination": {}},
    ]
    provider = ArizeAXTracingProvider(
        SimpleNamespace(experiments=experiments, datasets=datasets, spans=spans),
        experiment="experiment",
        score_name="quality",
    )
    (record,) = provider.iter_traces()
    assert record.metadata["reward"] == (0 if attempt_error else 0.75)
    assert len(record.spans) == 2
    assert datasets.list_examples.call_args.kwargs["cursor"] == "examples-next"
    assert spans.list.call_args.kwargs["cursor"] == "spans-next"
    assert spans.list.call_args.kwargs["project"] == "project"
    assert "trace-native" in spans.list.call_args.kwargs["filter"]


@pytest.mark.parametrize("attempt_error", [None, "Agent turn limit reached"])
def test_langfuse_experiment_uses_trace_and_item_score_subjects(attempt_error):
    experiments = create_autospec(LangfuseExperiments, instance=True)
    observations = create_autospec(ObservationsClient, instance=True)
    scores = create_autospec(ScoresV3Client, instance=True)
    experiments.list.return_value = {
        "data": [{"id": "experiment", "dataset_id": "dataset"}],
        "meta": {},
    }
    experiments.list_items.return_value = {
        "data": [
            {
                "id": "item-span",
                "trace_id": "trace",
                "experiment_id": "experiment",
                "experiment_item_id": "example",
                "experiment_item_version": "version",
                "level": "ERROR" if attempt_error else "DEFAULT",
            }
        ],
        "meta": {},
    }
    scores.get_many_v3.return_value = {
        "data": [
            {
                "id": "score",
                "name": "quality",
                "value": 0 if attempt_error else 0.5,
                "subject": {"kind": "observation", "id": "item-span", "trace_id": "trace"},
            },
            {
                "id": "other-score",
                "name": "quality",
                "value": 0,
                "subject": {"kind": "observation", "id": "other-span", "trace_id": "trace"},
            },
        ],
        "meta": {},
    }
    attrs = span_frame().to_dict(orient="records")[1]
    observations.get_many.return_value = {
        "data": [
            {
                "id": "llm",
                "trace_id": "trace",
                "metadata": {
                    "attributes": {
                        key.removeprefix("attributes."): value
                        for key, value in attrs.items()
                        if key.startswith("attributes.")
                    }
                },
            }
        ],
        "meta": {},
    }
    client = SimpleNamespace(
        get_dataset=Mock(
            return_value=SimpleNamespace(id="dataset", items=[SimpleNamespace(id="example")])
        ),
        api=SimpleNamespace(experiments=experiments, observations=observations, scores_v3=scores),
    )
    provider = LangfuseTracingProvider(
        client,
        dataset="dataset-name",
        experiment="experiment",
        from_start_time=datetime(2026, 1, 1, tzinfo=UTC),
        score_name="quality",
    )
    (record,) = provider.iter_traces()
    assert record.metadata["reward"] == (0 if attempt_error else 0.5)
    assert record.metadata["example_id"] == "example"
    assert observations.get_many.call_args.kwargs["trace_id"] == "trace"
    assert experiments.list_items.call_args.kwargs["dataset_id"] == "dataset"


@pytest.mark.parametrize("attempt_error", [None, "Agent turn limit reached"])
def test_langsmith_experiment_preserves_openinference_without_reconstructing_messages(
    attempt_error,
):
    client = create_autospec(LangSmithClient, instance=True)
    client.read_project.return_value = {
        "id": "experiment",
        "reference_dataset_id": "dataset",
        "extra": {"metadata": {"dataset_version": "version"}},
    }
    client.list_examples.return_value = [{"id": "example", "outputs": {"answer": "reference"}}]
    attrs = {
        "openinference.span.kind": "LLM",
        "llm.model_name": "policy",
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "go",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "north",
    }
    client.list_runs.side_effect = [
        [
            {
                "id": "root",
                "trace_id": "trace",
                "reference_example_id": "example",
                "error": attempt_error,
            }
        ],
        [{"id": "llm", "trace_id": "trace", "extra": {"metadata": attrs}}],
    ]
    client.list_feedback.return_value = [
        {"id": "score", "run_id": "root", "key": "quality", "score": 0 if attempt_error else 0.75}
    ]
    provider = LangSmithTracingProvider(client, experiment="experiment-name", score_name="quality")
    data = chat_dataset_from_tracing_provider(provider)
    assert data[0]["group_id"] == "example"
    assert data[0]["reward"] == (0 if attempt_error else 0.75)
    assert data[0]["model_id"] == "policy"
    assert json.loads(data[0]["metadata_json"])["experiment_id"] == "experiment"
    client.list_examples.assert_called_once_with(dataset_id="dataset", as_of="version")
    client.list_runs.assert_called_with(trace_id="trace")


@pytest.mark.parametrize(
    "evaluations",
    [
        [],
        [{"name": "quality", "score": 1}, {"name": "quality", "score": 0}],
        [{"name": "quality", "score": float("nan")}],
        [{"name": "quality", "score": "pass"}],
        [{"name": "quality", "score": 1, "error": "failed"}],
    ],
)
def test_missing_duplicate_or_invalid_selected_scores_fail(evaluations):
    with pytest.raises(ValueError):
        _selected_score(evaluations, "quality")


@pytest.mark.parametrize(
    "page",
    [
        {"data": [], "meta": {"cursor": "repeat"}},
        {"data": [], "pagination": {"has_more": True}},
    ],
)
def test_pagination_rejects_repeating_or_missing_continuations(page):
    with pytest.raises(ValueError, match="cursor"):
        list(cursor_items(Mock(return_value=page), {}))


def test_generated_ax_models_retain_read_only_ids_and_dynamic_trace_fields():
    run = ExperimentRun.from_dict({"id": "run", "example_id": "example", "trace_id": "trace"})
    value = native_mapping(run)
    assert value["id"] == "run" and value["trace_id"] == "trace"


def test_phoenix_async_experiment_joins_trace_and_reward():
    client = SimpleNamespace(
        experiments=SimpleNamespace(
            get_experiment=AsyncMock(
                return_value={
                    "experiment_id": "experiment",
                    "project_name": "project",
                    "dataset_id": "dataset",
                    "dataset_version_id": "version",
                    "task_runs": [
                        {"id": "run", "trace_id": "trace-native", "dataset_example_id": "example"}
                    ],
                    "evaluation_runs": [
                        {"experiment_run_id": "run", "name": "quality", "result": {"score": 1}}
                    ],
                }
            )
        ),
        datasets=SimpleNamespace(
            get_dataset=AsyncMock(
                return_value=PhoenixDataset(
                    {"id": "dataset", "name": "dataset"},
                    {"version_id": "version", "examples": [{"id": "example"}]},
                )
            )
        ),
        spans=SimpleNamespace(get_spans_dataframe=AsyncMock(return_value=span_frame())),
    )
    provider = ArizePhoenixTracingProvider(client, experiment="experiment", score_name="quality")

    async def collect():
        return [trace async for trace in provider.aiter_traces()]

    (trace,) = asyncio.run(collect())
    assert trace.metadata["example_id"] == "example" and trace.metadata["reward"] == 1
    client.datasets.get_dataset.assert_awaited_once_with(dataset="dataset", version_id="version")


@pytest.mark.parametrize("trace_id,span_trace", [("", "trace"), ("trace", "other-trace")])
def test_experiment_linkage_never_falls_back_to_reference_outputs(trace_id, span_trace):
    run = NativeExperimentRun(
        id="run",
        trace_id=trace_id,
        experiment_id="experiment",
        example_id="example",
        dataset_id="dataset",
        dataset_version_id="version",
        evaluations=[],
    )
    trace = TraceRecord(id=span_trace, spans=[{"SpanId": "llm"}])
    with pytest.raises(ValueError, match="missing linked OpenInference trace"):
        run.attach(trace, None)


def test_langfuse_refetches_flat_metadata_without_truncation(native_run):
    attrs = {
        "openinference.span.kind": "LLM",
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "go" * 300,
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "north",
    }
    observation = {"id": "llm", "trace_id": "trace", "metadata": attrs}
    truncated = {
        **observation,
        "metadata": {**attrs, "llm.input_messages.0.message.content": "go..."},
    }
    fetch = Mock(
        side_effect=[{"data": [truncated], "meta": {}}, {"data": [observation], "meta": {}}]
    )
    provider = LangfuseTracingProvider(
        SimpleNamespace(api=SimpleNamespace(observations=SimpleNamespace(get_many=fetch))),
        dataset="dataset",
        experiment="experiment",
        from_start_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    native_run.trace_id = "trace"
    provider.iter_experiment_runs = Mock(return_value=iter([native_run]))
    (trace,) = provider.iter_traces()
    (record,) = provider.spec.iter_chats(trace)
    assert record["messages"][0]["content"] == "go" * 300
    assert "llm.input_messages.0.message.content" in fetch.call_args.kwargs["expand_metadata"]


def test_experiment_cannot_be_replaced_by_a_dataset_reference():
    with pytest.raises(TypeError, match="experiment"):
        ArizePhoenixTracingProvider(SimpleNamespace(), dataset="dataset")


def test_duplicate_linked_traces_are_rejected():
    provider = ArizePhoenixTracingProvider(SimpleNamespace(), experiment="experiment")
    run = NativeExperimentRun(
        id="run",
        trace_id="trace",
        experiment_id="experiment",
        example_id="example",
        dataset_id="dataset",
        dataset_version_id="version",
        evaluations=[],
    )
    provider.iter_experiment_runs = Mock(return_value=iter([run, run]))
    provider._export_trace = Mock(
        return_value=iter([TraceRecord(id="trace", spans=[{"SpanId": "call"}])])
    )
    with pytest.raises(ValueError, match="share a trace"):
        list(provider.iter_traces())


def test_langfuse_generated_models_preserve_native_field_names():

    item = ExperimentItem(
        id="run",
        trace_id="trace",
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        level="DEFAULT",
        environment="sdk-experiment",
        experiment_id="experiment",
        experiment_name="test",
        experiment_item_id="example",
    )
    page = ExperimentItemsResponse(data=[item], meta={})
    value = native_mapping(page)["data"][0]
    assert value["trace_id"] == "trace"
    assert value["experiment_item_id"] == "example"
    assert isinstance(value["start_time"], str)


@pytest.mark.parametrize(
    "provider_type",
    [
        ArizePhoenixTracingProvider,
        ArizeAXTracingProvider,
        LangfuseTracingProvider,
        LangSmithTracingProvider,
    ],
)
@pytest.mark.parametrize(
    "removed", ["query", "dataset_query", "experiment_query", "reward", "group_id", "model_id"]
)
def test_only_native_experiments_can_select_provider_data(provider_type, removed):
    arguments = {"experiment": "experiment", removed: {}}
    if provider_type is LangfuseTracingProvider:
        arguments.update(dataset="dataset", from_start_time=datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(TypeError, match=removed):
        provider_type(SimpleNamespace(), **arguments)


def test_ax_linked_trace_uses_explicit_native_space(native_run):
    spans = Mock(return_value={"spans": [], "pagination": {}})
    provider = ArizeAXTracingProvider(
        SimpleNamespace(spans=SimpleNamespace(list=spans)),
        experiment="experiment",
        trace_project="my-project",
        space="my-space",
    )
    list(provider._export_trace(native_run))
    assert spans.call_args.kwargs == {
        "project": "my-project",
        "space": "my-space",
        "filter": "context.trace_id = 'trace-native'",
    }


def test_empty_experiment_is_rejected_before_provider_access():
    with pytest.raises(ValueError, match="native experiment"):
        ArizePhoenixTracingProvider(SimpleNamespace(), experiment=" ")


def test_langfuse_time_bound_requires_a_timezone():
    with pytest.raises(ValueError, match="timezone"):
        LangfuseTracingProvider(
            SimpleNamespace(),
            dataset="dataset",
            experiment="experiment",
            from_start_time=datetime(2026, 1, 1),
        )


def test_langfuse_mismatched_dataset_is_rejected_before_observation_export():
    client = SimpleNamespace(
        get_dataset=Mock(return_value=SimpleNamespace(id="wrong-dataset", items=[])),
        api=SimpleNamespace(
            experiments=SimpleNamespace(
                list=Mock(
                    return_value={
                        "data": [{"id": "experiment", "dataset_id": "dataset"}],
                        "meta": {},
                    }
                )
            )
        ),
    )
    provider = LangfuseTracingProvider(
        client,
        dataset="wrong-dataset",
        experiment="experiment",
        from_start_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="does not belong"):
        list(provider.iter_traces())
