"""Check assistant-only held-out loss and native dataset slicing."""

import math
from types import SimpleNamespace

import pytest

from run import _nll


def test_nll_ignores_context_and_normalizes_over_supervised_tokens():
    def tensor(data):
        return SimpleNamespace(data=data)

    batch = [SimpleNamespace(loss_fn_inputs={"weights": tensor([0, 0.25, 0.75])})]
    scored = SimpleNamespace(loss_fn_outputs=[{"logprobs": tensor([math.nan, -2, -4])}])
    calls = []

    def forward(data, loss):
        calls.append((data, loss))
        return SimpleNamespace(result=lambda: scored)

    assert _nll(SimpleNamespace(forward=forward), batch) == 3.5
    assert calls == [(batch, "cross_entropy")]


def test_nonfinite_action_logprob_is_not_a_valid_quality_measure():
    tensor = SimpleNamespace(data=[1])
    batch = [SimpleNamespace(loss_fn_inputs={"weights": tensor})]
    scored = SimpleNamespace(loss_fn_outputs=[{"logprobs": SimpleNamespace(data=[math.nan])}])
    client = SimpleNamespace(forward=lambda *args: SimpleNamespace(result=lambda: scored))
    with pytest.raises(ValueError, match="nonfinite"):
        _nll(client, batch)
