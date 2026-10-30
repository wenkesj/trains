from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from datasets import Dataset
from langfuse import Langfuse

import claude_tracing.langfuse as module
from claude_tracing.training import Policy


@pytest.mark.parametrize(
    ("trace_ids", "span_ids", "message"),
    [
        (["trace-a"], ["call-a"], "omitted an attempt"),
        (["trace-a", "trace-b"], ["call-a", "wrong-call"], "exact recorded"),
        (["trace-a", "trace-b", "trace-b"], ["call-a", "call-b", "call-b"], "exact recorded"),
        (["trace-a", "trace-b"], ["call-a", "call-b"], None),
    ],
)
def test_langfuse_export_requires_every_original_attempt_and_exact_sdk_call(
    monkeypatch, trace_ids, span_ids, message
):
    provider = SimpleNamespace(
        iter_experiment_runs=lambda: [
            SimpleNamespace(evaluations=[1]),
            SimpleNamespace(evaluations=[0]),
        ]
    )
    monkeypatch.setattr(module, "LangfuseTracingProvider", lambda *args, **kwargs: provider)
    chats = Dataset.from_dict({"trace_id": trace_ids, "span_id": span_ids})
    monkeypatch.setattr(module, "chat_dataset_from_tracing_provider", lambda provider: chats)
    receipt = {
        "experiment_id": "experiment",
        "dataset": "dataset",
        "from_start_time": "2026-10-30T00:00:00+00:00",
        "attempts": [
            {"trace_id": "trace-a", "model_call_span_ids": ["call-a"]},
            {"trace_id": "trace-b", "model_call_span_ids": ["call-b"]},
        ],
    }
    client = Mock(
        spec=Langfuse,
        api=SimpleNamespace(
            experiments=SimpleNamespace(list=lambda **kwargs: {"data": [{"id": "experiment"}]})
        ),
    )
    backend = module.LangfuseBackend(client)
    policy = Mock(spec=Policy, rendering_options={})
    if message:
        with pytest.raises(ValueError, match=message):
            backend.chats(receipt, policy)
    else:
        assert backend.chats(receipt, policy) is chats


def test_langfuse_waits_for_experiment_indexing_before_reading_its_runs(monkeypatch):
    read_runs = Mock(return_value=[SimpleNamespace(evaluations=[0])])
    provider = SimpleNamespace(iter_experiment_runs=read_runs)
    monkeypatch.setattr(module, "LangfuseTracingProvider", lambda *args, **kwargs: provider)
    chats = Dataset.from_dict({"trace_id": ["trace"], "span_id": ["call"]})
    monkeypatch.setattr(module, "chat_dataset_from_tracing_provider", lambda provider: chats)
    sleep = Mock()
    monkeypatch.setattr(module.time, "sleep", sleep)
    list_experiments = Mock(side_effect=[{"data": []}, {"data": [{"id": "experiment"}]}])
    client = Mock(
        spec=Langfuse, api=SimpleNamespace(experiments=SimpleNamespace(list=list_experiments))
    )
    receipt = {
        "experiment_id": "experiment",
        "dataset": "dataset",
        "from_start_time": "2026-10-30T00:00:00+00:00",
        "attempts": [{"trace_id": "trace", "model_call_span_ids": ["call"]}],
    }
    result = module.LangfuseBackend(client).chats(receipt, Mock(spec=Policy, rendering_options={}))
    assert result is chats
    assert list_experiments.call_count == 2
    assert read_runs.call_count == 1
    sleep.assert_called_once_with(1)
