from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from datasets import Dataset

from opencode_tracing.langsmith import LangSmithBackend


def test_export_waits_for_failed_attempt_and_zero_score_to_be_visible(monkeypatch):
    successful = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000001",
        name="ChatCompletion",
        error=None,
        extra={"metadata": {"OTEL_TRACE_ID": "otel-a"}},
    )
    failed = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000002",
        name="ChatCompletion",
        error=None,
        extra={"metadata": {"OTEL_TRACE_ID": "otel-b"}},
    )
    roots = iter([[successful], [successful, failed]])

    def indexed_runs(**kwargs):
        if kwargs.get("is_root"):
            return next(roots)
        return [successful if "otel-a" in kwargs["filter"] else failed]

    client = SimpleNamespace(
        list_runs=Mock(side_effect=indexed_runs),
        list_feedback=Mock(
            return_value=[
                SimpleNamespace(run_id="00000000-0000-0000-0000-000000000001", score=1),
                SimpleNamespace(run_id="00000000-0000-0000-0000-000000000002", score=0),
            ]
        ),
    )
    sleeps = []
    monkeypatch.setattr("opencode_tracing.langsmith.time.sleep", sleeps.append)
    backend = LangSmithBackend(client, project="traces")
    backend._await_experiment(
        {
            "experiment_id": "experiment",
            "attempts": [
                {"trace_id": "otel-a", "model_call_span_ids": ["0000000000000001"]},
                {"trace_id": "otel-b", "model_call_span_ids": ["0000000000000002"]},
            ],
        }
    )
    assert client.list_runs.call_count == 4
    assert len(sleeps) == 1


def test_partial_export_cannot_proceed_to_ppo(monkeypatch):
    backend = LangSmithBackend(SimpleNamespace(), project="traces")
    monkeypatch.setattr(backend, "_await_experiment", lambda receipt: None)
    monkeypatch.setattr(
        "opencode_tracing.langsmith.chat_dataset_from_tracing_provider",
        lambda provider: Dataset.from_list([{"trace_id": "native-a"}]),
    )
    with pytest.raises(ValueError, match="omitted an attempt"):
        backend.chats(
            {
                "experiment_id": "experiment",
                "dataset": "dataset",
                "attempts": [
                    {"trace_id": "otel-a", "model_call_span_ids": ["0000000000000001"]},
                    {"trace_id": "otel-b", "model_call_span_ids": ["0000000000000002"]},
                ],
            },
            SimpleNamespace(rendering_options={}),
        )


def test_export_also_waits_for_linked_model_trace_index(monkeypatch):
    run = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000001",
        name="ChatCompletion",
        error=None,
        extra={"metadata": {"OTEL_TRACE_ID": "otel-a"}},
    )
    client = SimpleNamespace(
        list_runs=Mock(side_effect=[[run], [], [run], [run]]),
        list_feedback=Mock(
            return_value=[SimpleNamespace(run_id="00000000-0000-0000-0000-000000000001", score=0)]
        ),
    )
    sleeps = []
    monkeypatch.setattr("opencode_tracing.langsmith.time.sleep", sleeps.append)
    LangSmithBackend(client, project="traces")._await_experiment(
        {
            "experiment_id": "experiment",
            "attempts": [{"trace_id": "otel-a", "model_call_span_ids": ["0000000000000001"]}],
        }
    )
    assert len(sleeps) == 1
    assert client.list_runs.call_count == 4


def test_export_waits_for_every_completed_sdk_call(monkeypatch):
    root = SimpleNamespace(id="attempt", extra={"metadata": {"OTEL_TRACE_ID": "otel-a"}})
    first = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000001", name="ChatCompletion", error=None
    )
    second = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000002", name="messages.create", error=None
    )
    client = SimpleNamespace(
        list_runs=Mock(side_effect=[[root], [first], [root], [first, second]]),
        list_feedback=Mock(return_value=[SimpleNamespace(run_id="attempt", score=0)]),
    )
    sleeps = []
    monkeypatch.setattr("opencode_tracing.langsmith.time.sleep", sleeps.append)
    LangSmithBackend(client, project="traces")._await_experiment(
        {
            "experiment_id": "experiment",
            "attempts": [
                {
                    "trace_id": "otel-a",
                    "model_call_span_ids": ["0000000000000001", "0000000000000002"],
                }
            ],
        }
    )
    assert len(sleeps) == 1
    assert client.list_runs.call_count == 4


def test_partial_model_calls_cannot_proceed_to_ppo(monkeypatch):
    backend = LangSmithBackend(SimpleNamespace(), project="traces")
    monkeypatch.setattr(backend, "_await_experiment", lambda receipt: None)
    monkeypatch.setattr(
        "opencode_tracing.langsmith.chat_dataset_from_tracing_provider",
        lambda provider: Dataset.from_list([{"trace_id": "native-a"}]),
    )
    with pytest.raises(ValueError, match="omitted model calls"):
        backend.chats(
            {
                "experiment_id": "experiment",
                "dataset": "dataset",
                "attempts": [
                    {
                        "trace_id": "otel-a",
                        "model_call_span_ids": ["0000000000000001", "0000000000000002"],
                    }
                ],
            },
            SimpleNamespace(rendering_options={}),
        )


def test_completed_generation_with_error_status_is_counted():
    root = SimpleNamespace(id="attempt", extra={"metadata": {"OTEL_TRACE_ID": "otel-a"}})
    completed = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000001",
        name="Response",
        error="tool failed",
        outputs={
            "object": "response",
            "status": "completed",
            "error": None,
            "output": [{"type": "function_call", "status": "completed"}],
        },
    )
    client = SimpleNamespace(
        list_runs=Mock(side_effect=[[root], [completed]]),
        list_feedback=Mock(return_value=[SimpleNamespace(run_id="attempt", score=0)]),
    )
    LangSmithBackend(client, project="traces")._await_experiment(
        {
            "experiment_id": "experiment",
            "attempts": [{"trace_id": "otel-a", "model_call_span_ids": ["0000000000000001"]}],
        }
    )


def test_matching_call_count_with_wrong_identity_is_rejected():
    root = SimpleNamespace(id="attempt", extra={"metadata": {"OTEL_TRACE_ID": "otel-a"}})
    wrong = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000002", name="ChatCompletion", error=None
    )
    client = SimpleNamespace(
        list_runs=Mock(side_effect=[[root], [wrong]]),
        list_feedback=Mock(return_value=[SimpleNamespace(run_id="attempt", score=0)]),
    )
    with pytest.raises(ValueError, match="unrecorded SDK"):
        LangSmithBackend(client, project="traces")._await_experiment(
            {
                "experiment_id": "experiment",
                "attempts": [{"trace_id": "otel-a", "model_call_span_ids": ["0000000000000001"]}],
            }
        )


def test_exported_call_identity_is_checked_after_provider_conversion(monkeypatch):
    backend = LangSmithBackend(SimpleNamespace(), project="traces")
    monkeypatch.setattr(backend, "_await_experiment", lambda receipt: None)
    monkeypatch.setattr(
        "opencode_tracing.langsmith.chat_dataset_from_tracing_provider",
        lambda provider: Dataset.from_list(
            [{"trace_id": "native-a", "span_id": "00000000-0000-0000-0000-000000000002"}]
        ),
    )
    with pytest.raises(ValueError, match="changed model call identities"):
        backend.chats(
            {
                "experiment_id": "experiment",
                "dataset": "dataset",
                "attempts": [{"trace_id": "otel-a", "model_call_span_ids": ["0000000000000001"]}],
            },
            SimpleNamespace(rendering_options={}),
        )
