"""Verify repeated CLI rounds consume the newly published policy."""

import json
import sys
from types import SimpleNamespace

import pytest

import run


def test_rl_publishes_policy_used_by_next_collect(tmp_path, monkeypatch, capsys):
    original = {
        "model": "bt://before",
        "state": "bt://before-state",
        "base_model": "LiquidAI/LFM3-Omni-2B",
    }
    updated = {
        **original,
        "model": "bt://after",
        "state": "bt://after-state",
        "rl_metrics": {"updates": 3},
    }
    (tmp_path / "policy.json").write_text(json.dumps(original))
    collection = tmp_path / "previous-collection"
    collection.mkdir()
    calls = []
    sampler = SimpleNamespace(close=lambda: calls.append("closed"))

    def sampling_client(*, model_path):
        assert model_path == updated["model"]
        return sampler

    service = SimpleNamespace(create_sampling_client=sampling_client)
    monkeypatch.setattr(run.loops, "ServiceClient", lambda: service)

    def reinforce(path, policy, client, sdk, *, steps, learning_rate):
        assert path == collection and policy == original and client is service
        assert steps == 3 and learning_rate == 5e-5
        return updated

    monkeypatch.setattr(run, "reinforce", reinforce)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "rl",
            "--artifacts",
            str(tmp_path),
            "--collection",
            str(collection),
            "--steps",
            "3",
            "--learning-rate",
            "5e-5",
        ],
    )
    run.main()
    assert json.loads((tmp_path / "policy.json").read_text()) == updated
    assert not (tmp_path / "policy.json.tmp").exists()
    assert json.loads(capsys.readouterr().out) == updated

    monkeypatch.setattr(run, "load_renderer", lambda model: object())
    monkeypatch.setattr(run.Exercise, "load", lambda repository, slug: SimpleNamespace(slug=slug))

    def collect(artifacts, sessions, policy, actual_sampler, renderer, **kwargs):
        assert artifacts == tmp_path and policy == updated and actual_sampler is sampler
        assert len(kwargs["exercises"]) == 1
        calls.append("collected")
        return tmp_path / "next-collection"

    monkeypatch.setattr(run, "collect", collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "collect",
            "--artifacts",
            str(tmp_path),
            "--exercises",
            "two-fer",
            "--attempts",
            "2",
        ],
    )
    run.main()
    assert calls == ["collected", "closed"]


@pytest.mark.parametrize("rate", ["0", "-1", "nan", "inf"])
def test_invalid_learning_rate_fails_before_creating_a_service(rate, monkeypatch):
    def unexpected_service():
        pytest.fail("Invalid CLI parameters must fail before allocating a service")

    monkeypatch.setattr(run.loops, "ServiceClient", unexpected_service)
    monkeypatch.setattr(sys, "argv", ["run.py", "sft", "--learning-rate", rate])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
