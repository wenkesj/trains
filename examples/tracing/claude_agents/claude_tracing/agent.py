# Run the native Claude Agent SDK, with its upstream agent/tool instrumentor.

import asyncio
import os
from typing import Any

import claude_agent_sdk
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ResultError,
    ResultMessage,
    create_sdk_mcp_server,
    tool,
)

from claude_tracing.model_proxy import ModelProxy
from claude_tracing.result import AgentResult
from claude_tracing.sandbox import Sandbox
from claude_tracing.training import Policy


async def solve(sandbox: Sandbox, policy: Policy, *, max_turns: int = 20) -> AgentResult:
    """Run the native coding harness against the candidate workspace and policy."""

    @tool(
        "execute",
        "Run a shell command in the isolated exercise workspace to read, edit, and test Python files.",
        {"command": str},
    )
    async def execute(arguments: dict[str, Any]) -> dict:
        result = await asyncio.to_thread(sandbox.execute, arguments["command"])
        return {"content": [{"type": "text", "text": result}]}

    server = create_sdk_mcp_server("sandbox", tools=[execute])
    with ModelProxy(policy) as proxy:
        options = ClaudeAgentOptions(
            model=policy.model,
            tools=[],
            allowed_tools=["mcp__sandbox__execute"],
            mcp_servers={"sandbox": server},
            strict_mcp_config=True,
            permission_mode="dontAsk",
            setting_sources=[],
            skills=[],
            system_prompt="Implement the Python exercise in /workspace. Read INSTRUCTIONS.md and the starter solution files with the sandbox execute tool. Implement the required functions without changing their signatures. After checking one representative example, finish with a concise summary. Hidden upstream tests are run separately. Do not ask for user input.",
            max_turns=max_turns,
            thinking={"type": "disabled"},
            env={
                "ANTHROPIC_BASE_URL": proxy.url,
                "ANTHROPIC_AUTH_TOKEN": os.environ["TINKER_API_KEY"],
                "ANTHROPIC_API_KEY": os.environ["TINKER_API_KEY"],
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "2048",
            },
        )
        result = None
        exhausted = False
        try:
            async for message in claude_agent_sdk.query(
                prompt="Read INSTRUCTIONS.md and implement the exercise.", options=options
            ):
                if isinstance(message, ResultMessage):
                    if message.subtype == "error_max_turns":
                        exhausted = True
                    elif message.is_error:
                        raise RuntimeError(f"Claude Agent SDK did not complete: {message.subtype}")
                    result = message.result
        except ResultError as error:
            if error.subtype != "error_max_turns":
                raise
            exhausted = True
        if exhausted:
            return AgentResult(None, "max_turns")
        if result is None:
            raise RuntimeError("Claude Agent SDK returned no final result")
        return AgentResult(result, "completed")
