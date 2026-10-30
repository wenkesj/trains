# Run the actual OpenAI Agents SDK against the published Tinker policy.

import asyncio
import os

from agents import Agent, ModelSettings, OpenAIResponsesModel, Runner, function_tool
from agents.exceptions import MaxTurnsExceeded
from openai import AsyncOpenAI

from openai_tracing.result import AgentResult
from openai_tracing.sandbox import Sandbox
from openai_tracing.training import Policy


async def solve(sandbox: Sandbox, policy: Policy, *, max_turns: int = 20) -> AgentResult:
    """Run the native coding harness against the candidate workspace and policy."""

    @function_tool
    async def execute(command: str) -> str:
        """Run a shell command in the isolated exercise workspace; use it to read and edit files."""
        return await asyncio.to_thread(sandbox.execute, command)

    client = AsyncOpenAI(base_url=policy.inference_url, api_key=os.environ["TINKER_API_KEY"])
    try:
        agent = Agent(
            name="Exercism Python developer",
            instructions="Implement the exercise in /workspace. Read INSTRUCTIONS.md and the starter Python files, then edit those solution files. You may execute Python to test your own examples. Hidden upstream tests run after you finish. Do not change the required function signatures. After implementing the solution and checking one representative example, finish with a concise summary. Complete the work without asking for input.",
            model=OpenAIResponsesModel(model=policy.model, openai_client=client),
            model_settings=ModelSettings(
                temperature=1,
                top_p=1,
                max_tokens=2048,
                extra_body={"chat_template_kwargs": policy.rendering_options},
            ),
            tools=[execute],
        )
        try:
            result = await Runner.run(
                agent, "Read INSTRUCTIONS.md and implement the exercise.", max_turns=max_turns
            )
        except MaxTurnsExceeded:
            return AgentResult(None, "max_turns")
        return AgentResult(str(result.final_output), "completed")
    finally:
        await client.close()
