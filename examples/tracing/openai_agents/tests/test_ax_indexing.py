from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from datasets import Dataset

from openai_tracing.arize_ax import AXBackend


def span(identifier):
    return {
        "context.trace_id": "trace",
        "context.span_id": identifier,
        "name": "ChatCompletion",
        "status_code": "OK",
        "attributes": {"openinference.span.kind": "LLM"},
    }


def receipt():
    return {
        "experiment_id": "experiment",
        "dataset_id": "dataset",
        "trace_project": "project",
        "space": "space",
        "attempts": [{"trace_id": "trace", "model_call_span_ids": ["call-a", "call-b"]}],
    }


def test_export_waits_for_exact_sdk_call_ids(monkeypatch):
    spans = SimpleNamespace(
        list=Mock(
            side_effect=[{"spans": [span("call-a")]}, {"spans": [span("call-a"), span("call-b")]}]
        )
    )
    backend = AXBackend(SimpleNamespace(spans=spans), project="project", space="space")
    sleeps = []
    monkeypatch.setattr("openai_tracing.arize_ax.time.sleep", sleeps.append)
    assert backend._await_calls(receipt()) == {"call-a", "call-b"}
    assert len(sleeps) == 1


def test_unrecorded_call_is_not_accepted_in_place_of_a_missing_call():
    spans = SimpleNamespace(list=Mock(return_value={"spans": [span("call-a"), span("unexpected")]}))
    backend = AXBackend(SimpleNamespace(spans=spans), project="project", space="space")
    with pytest.raises(ValueError, match="unrecorded"):
        backend._await_calls(receipt())


def test_partial_export_cannot_reach_ppo(monkeypatch):
    backend = AXBackend(SimpleNamespace(), project="project", space="space")
    monkeypatch.setattr(backend, "_await_calls", lambda receipt: {"call-a", "call-b"})
    monkeypatch.setattr(
        "openai_tracing.arize_ax.chat_dataset_from_tracing_provider",
        lambda provider: Dataset.from_list([{"span_id": "call-a"}]),
    )
    with pytest.raises(ValueError, match="omitted model calls"):
        backend.chats(receipt(), SimpleNamespace(rendering_options={}))


def test_explicit_native_completion_overrides_error_span_status():
    completed = span("call-b")
    completed["status_code"] = "ERROR"
    completed["name"] = "Response"
    completed["attributes"]["output.value"] = {
        "object": "response",
        "status": "completed",
        "error": None,
        "output": [{"type": "function_call", "status": "completed"}],
    }
    spans = SimpleNamespace(list=Mock(return_value={"spans": [span("call-a"), completed]}))
    backend = AXBackend(SimpleNamespace(spans=spans), project="project", space="space")
    assert backend._await_calls(receipt()) == {"call-a", "call-b"}
