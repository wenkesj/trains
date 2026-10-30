from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trains.datasets import chat_dataset_from_tracing_provider
from trains.formats.tracing import TraceSpec
from trains.tracing import LangSmithTracingProvider


def test_langsmith_lazy_export_ignores_non_model_runs(processor, native_run):
    records = [
        {
            "trace_id": "trace",
            "id": "root",
            "extra": {"metadata": {"openinference.span.kind": "CHAIN"}},
        },
        {
            "trace_id": "trace",
            "id": "llm",
            "parent_run_id": "root",
            "run_type": "llm",
            "extra": {
                "metadata": {
                    "openinference.span.kind": "LLM",
                    "llm.input_messages.0.message.role": "user",
                    "llm.input_messages.0.message.content": "go",
                    "llm.output_messages.0.message.role": "assistant",
                    "llm.output_messages.0.message.content": "north",
                }
            },
        },
    ]
    client = SimpleNamespace(list_runs=Mock(return_value=iter(records)))
    native_run.trace_id = "trace"
    provider = LangSmithTracingProvider(client, TraceSpec(), experiment="experiment")
    provider.iter_experiment_runs = Mock(return_value=iter([native_run]))
    data = chat_dataset_from_tracing_provider(provider)
    assert len(data) == 1 and data[0]["trace_id"] == "trace"


def test_langsmith_native_otlp_runs_restore_consumed_openinference_fields(native_run):
    records = [
        {
            "trace_id": "trace",
            "id": "call",
            "name": "ChatCompletion",
            "run_type": "llm",
            "inputs": {
                "model": "policy",
                "messages": [{"role": "user", "content": "Solve it"}],
                "temperature": 1,
                "top_p": 1,
            },
            "outputs": {
                "model": "policy",
                "choices": [
                    {"message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}
                ],
            },
            "extra": {
                "metadata": {
                    "OTEL_TRACE_ID": "original-otel-id",
                    "ls_model_name": "policy",
                    "otel.span.input.mime_type": "application/json",
                }
            },
        }
    ]
    client = SimpleNamespace(list_runs=Mock(return_value=iter(records)))
    native_run.trace_id = "trace"
    provider = LangSmithTracingProvider(client, TraceSpec(), experiment="experiment")
    provider.iter_experiment_runs = Mock(return_value=iter([native_run]))
    data = chat_dataset_from_tracing_provider(provider)
    assert len(data) == 1
    assert data[0]["messages"][-1]["content"] == "Answer"
    assert data[0]["model_id"] == "policy"


def test_langsmith_experiment_links_otel_metadata_to_provider_native_trace_identity():
    def list_runs(**kwargs):
        if "project_id" in kwargs:
            return iter(
                [
                    {
                        "id": "execution",
                        "trace_id": "experiment-root",
                        "reference_example_id": "task",
                        "extra": {"metadata": {"OTEL_TRACE_ID": "original-otel-id"}},
                    }
                ]
            )
        assert kwargs["project_name"] == "trace-project"
        if "filter" in kwargs:
            assert '"OTEL_TRACE_ID"' in kwargs["filter"]
            assert '"original-otel-id"' in kwargs["filter"]
            return iter(
                [
                    {
                        "trace_id": "provider-native-root",
                        "extra": {"metadata": {"OTEL_TRACE_ID": "original-otel-id"}},
                    }
                ]
            )
        assert kwargs["trace_id"] == "provider-native-root"
        return iter(
            [
                {
                    "id": "model-call",
                    "trace_id": "provider-native-root",
                    "name": "ChatCompletion",
                    "run_type": "llm",
                    "inputs": {"messages": [{"role": "user", "content": "Solve"}]},
                    "outputs": {"role": "assistant", "content": "Incorrect answer"},
                }
            ]
        )

    client = SimpleNamespace(
        read_project=lambda **kwargs: {"id": "experiment", "reference_dataset_id": "dataset"},
        list_examples=lambda **kwargs: [{"id": "task"}],
        list_runs=list_runs,
        list_feedback=lambda **kwargs: [
            {"id": "grade", "run_id": "execution", "key": "score", "score": 0}
        ],
    )
    data = chat_dataset_from_tracing_provider(
        LangSmithTracingProvider(
            client, experiment="experiment", trace_project="trace-project", score_name="score"
        )
    )
    assert len(data) == 1
    assert data[0]["trace_id"] == "provider-native-root"
    assert data[0]["reward"] == 0
    assert data[0]["messages"][-1]["content"] == "Incorrect answer"


def test_native_anthropic_provider_restores_tool_messages_before_training(native_run):
    records = [
        {
            "id": "anthropic-call",
            "trace_id": "trace",
            "run_type": "llm",
            "name": "messages.create",
            "inputs": {
                "model": "policy",
                "system": "Implement the exercise.",
                "tools": [{"name": "execute", "input_schema": {"type": "object"}}],
                "messages": [
                    {"role": "user", "content": "Solve it"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool-1",
                                "name": "execute",
                                "input": {"command": "pwd"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-1",
                                "content": [{"type": "text", "text": "/workspace"}],
                            }
                        ],
                    },
                ],
            },
            "outputs": {"role": "assistant", "content": [{"type": "text", "text": "Done"}]},
            "extra": {"metadata": {"ls_provider": "anthropic", "ls_model_name": "policy"}},
        }
    ]
    client = SimpleNamespace(list_runs=Mock(return_value=iter(records)))
    native_run.trace_id = "trace"
    provider = LangSmithTracingProvider(client, TraceSpec(), experiment="experiment")
    provider.iter_experiment_runs = Mock(return_value=iter([native_run]))
    row = chat_dataset_from_tracing_provider(provider)[0]
    assert [m["role"] for m in row["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert row["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"command": "pwd"}
    assert row["messages"][2]["weight"] == 0
    assert row["messages"][3]["content"] == "/workspace"
    assert row["messages"][-1]["content"] == "Done"


def test_feedback_batches_preserve_each_attempt_score_and_exclude_foreign_runs():
    roots = [
        {"id": "run-a", "trace_id": "trace-a", "reference_example_id": "a"},
        {"id": "foreign", "trace_id": "foreign-trace", "reference_example_id": "other"},
        {"id": "run-b", "trace_id": "trace-b", "reference_example_id": "b", "error": "failed"},
    ]
    client = SimpleNamespace(
        read_project=lambda **kwargs: {"id": "experiment", "reference_dataset_id": "dataset"},
        list_examples=lambda **kwargs: [{"id": "a"}, {"id": "b"}],
        list_runs=lambda **kwargs: iter(roots),
        list_feedback=Mock(
            return_value=iter(
                [
                    {"id": "grade-b", "run_id": "run-b", "key": "score", "score": 0},
                    {"id": "foreign-grade", "run_id": "foreign", "key": "score", "score": 1},
                    {"id": "grade-a", "run_id": "run-a", "key": "score", "score": 0.75},
                ]
            )
        ),
    )
    runs = list(
        LangSmithTracingProvider(
            client, experiment="experiment", score_name="score"
        ).iter_experiment_runs()
    )
    assert [run.id for run in runs] == ["run-a", "run-b"]
    assert [run.evaluations for run in runs] == [
        [{"id": "grade-a", "name": "score", "score": 0.75}],
        [{"id": "grade-b", "name": "score", "score": 0}],
    ]
    client.list_feedback.assert_called_once_with(run_ids=["run-a", "run-b"], feedback_key=["score"])


def test_feedback_reads_are_bounded_for_large_experiments():
    roots = [
        {"id": f"run-{i}", "trace_id": f"trace-{i}", "reference_example_id": "a"} for i in range(65)
    ]

    def feedback(**kwargs):
        return iter(
            {"id": run_id + "-grade", "run_id": run_id, "key": "score", "score": 0}
            for run_id in kwargs["run_ids"]
        )

    client = SimpleNamespace(
        read_project=lambda **kwargs: {"id": "experiment", "reference_dataset_id": "dataset"},
        list_examples=lambda **kwargs: [{"id": "a"}],
        list_runs=lambda **kwargs: iter(roots),
        list_feedback=Mock(side_effect=feedback),
    )
    runs = list(
        LangSmithTracingProvider(
            client, experiment="experiment", score_name="score"
        ).iter_experiment_runs()
    )
    assert len(runs) == 65
    assert [len(call.kwargs["run_ids"]) for call in client.list_feedback.call_args_list] == [64, 1]
    assert all(run.evaluations[0]["score"] == 0 for run in runs)


def test_trace_links_are_batched_without_mixing_attempt_identities():
    roots = [
        {
            "id": "a",
            "trace_id": "experiment-a",
            "reference_example_id": "example",
            "extra": {"metadata": {"OTEL_TRACE_ID": "otel-a"}},
        },
        {
            "id": "b",
            "trace_id": "experiment-b",
            "reference_example_id": "example",
            "extra": {"metadata": {"OTEL_TRACE_ID": "otel-b"}},
        },
    ]
    linked = [
        {"trace_id": "native-b", "extra": {"metadata": {"OTEL_TRACE_ID": "otel-b"}}},
        {"trace_id": "native-a", "extra": {"metadata": {"OTEL_TRACE_ID": "otel-a"}}},
        {"trace_id": "native-a", "extra": {"metadata": {"OTEL_TRACE_ID": "otel-a"}}},
    ]
    client = SimpleNamespace(
        read_project=lambda **kwargs: {"id": "experiment", "reference_dataset_id": "dataset"},
        list_examples=lambda **kwargs: [{"id": "example"}],
        list_runs=Mock(side_effect=[iter(roots), iter(linked)]),
    )
    runs = list(
        LangSmithTracingProvider(
            client, experiment="experiment", trace_project="traces"
        ).iter_experiment_runs()
    )
    assert [(run.id, run.trace_id) for run in runs] == [("a", "native-a"), ("b", "native-b")]
    assert client.list_runs.call_count == 2
    kwargs = client.list_runs.call_args.kwargs
    assert kwargs["project_name"] == "traces"
    assert (
        "in(metadata_value," in kwargs["filter"]
        and '"otel-a"' in kwargs["filter"]
        and '"otel-b"' in kwargs["filter"]
    )
    assert kwargs["select"] == ["id", "trace_id", "extra"]


@pytest.mark.parametrize("native_ids", [[], ["native-a", "native-b"]])
def test_missing_or_ambiguous_native_trace_links_fail(native_ids):
    root = {
        "id": "a",
        "trace_id": "experiment-a",
        "reference_example_id": "example",
        "extra": {"metadata": {"OTEL_TRACE_ID": "otel-a"}},
    }
    client = SimpleNamespace(
        read_project=lambda **kwargs: {"id": "experiment", "reference_dataset_id": "dataset"},
        list_examples=lambda **kwargs: [{"id": "example"}],
        list_runs=Mock(
            side_effect=[
                [root],
                [
                    {"trace_id": trace_id, "extra": {"metadata": {"OTEL_TRACE_ID": "otel-a"}}}
                    for trace_id in native_ids
                ],
            ]
        ),
    )
    with pytest.raises(ValueError, match="missing or ambiguous"):
        list(
            LangSmithTracingProvider(
                client, experiment="experiment", trace_project="traces"
            ).iter_experiment_runs()
        )
