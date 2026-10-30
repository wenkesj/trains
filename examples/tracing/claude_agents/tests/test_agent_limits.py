"""Retain native turn-limit attempts while surfacing infrastructure failures."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from claude_agent_sdk import ResultError

from claude_tracing import agent


@pytest.mark.parametrize("subtype", ["error_max_turns", "error_during_execution"])
def test_terminal_sdk_errors_distinguish_turn_limits_from_failed_requests(monkeypatch, subtype):
    error = ResultError("native terminal result", data={"subtype": subtype}, exit_code=1)

    async def query(**kwargs):
        if False:
            yield
        raise error

    monkeypatch.setenv("TINKER_API_KEY", "tml-local")
    monkeypatch.setattr(agent.claude_agent_sdk, "query", query)
    monkeypatch.setattr(
        agent, "ModelProxy", lambda policy: nullcontext(SimpleNamespace(url="http://127.0.0.1"))
    )
    run = agent.solve(Mock(spec=agent.Sandbox), Mock(spec=agent.Policy, model="local-policy"))
    if subtype == "error_max_turns":
        result = asyncio.run(run)
        assert result.stop_reason == "max_turns" and result.response is None
    else:
        with pytest.raises(ResultError):
            asyncio.run(run)
