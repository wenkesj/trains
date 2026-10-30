import asyncio
import json

from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool

from deepagents_tracing.agent import _canonical_tools


def test_middleware_orders_model_schema_and_retains_the_executable_tool():
    @tool
    def execute(z: str, a: str) -> str:
        """Execute a command in a directory."""
        return z + a

    request = ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        tools=[execute],
        messages=[],
        state={"messages": []},
    )
    captured = {}

    async def handler(bound):
        captured["request"] = bound
        return "native-response"

    result = asyncio.run(_canonical_tools.awrap_model_call(request, handler))
    bound = captured["request"]
    assert result == "native-response"
    assert request.tools == [execute]
    assert execute.invoke({"z": "command", "a": "directory"}) == "commanddirectory"
    assert bound.messages is request.messages
    assert bound.tools[0]["function"]["name"] == "execute"
    assert json.dumps(bound.tools) == json.dumps(
        json.loads(json.dumps(bound.tools, sort_keys=True))
    )
