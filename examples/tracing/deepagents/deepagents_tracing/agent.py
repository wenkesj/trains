# Deep Agents with native tools operating in the candidate Docker container.

import json
import os
import subprocess
from collections.abc import Awaitable, Callable
from typing import Any

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, wrap_model_call
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from recipes.tracing import canonical_tool_schemas

from deepagents_tracing.result import AgentResult
from deepagents_tracing.sandbox import Sandbox
from deepagents_tracing.training import Policy


class DockerBackend(BaseSandbox):
    """Use Deep Agents' filesystem implementation without exposing host files."""

    def __init__(self, sandbox: Sandbox) -> None:
        """Expose only the running candidate container to Deep Agents tools.

        Args:
            sandbox: Running candidate container used for all filesystem and shell operations.
        """
        self.sandbox = sandbox

    @property
    def id(self) -> str:
        """Return the running candidate container identity."""
        if self.sandbox.container is None:
            raise RuntimeError("Sandbox is not running")
        return self.sandbox.container

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a shell command in the candidate container with a finite timeout."""
        result = json.loads(self.sandbox.execute(command, timeout=timeout or 30))
        return ExecuteResponse(
            output=result["stdout"] + result["stderr"],
            exit_code=result["exit_code"],
            truncated=result["truncated"],
        )

    def _transfer(
        self, path: str, *, content: bytes | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        operation = (
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())"
            if content is not None
            else "import pathlib,sys; sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())"
        )
        return subprocess.run(
            ["docker", "exec", "-i", self.id, "python3", "-c", operation, path],
            input=content,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Write binary files inside the candidate container without shell interpolation."""
        return [
            FileUploadResponse(
                path=path,
                error="invalid_path" if self._transfer(path, content=data).returncode else None,
            )
            for path, data in files
        ]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Read binary files from the candidate container without exposing host paths."""
        responses = []
        for path in paths:
            result = self._transfer(path)
            responses.append(
                FileDownloadResponse(
                    path=path,
                    content=result.stdout if not result.returncode else None,
                    error="file_not_found" if result.returncode else None,
                )
            )
        return responses


async def solve(sandbox: Sandbox, policy: Policy, *, max_turns: int = 20) -> AgentResult:
    """Run the native coding harness against the candidate workspace and policy."""
    register_harness_profile(
        "openai",
        HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
    )
    model = ChatOpenAI(
        model=policy.model,
        base_url=policy.inference_url,
        api_key=os.environ["TINKER_API_KEY"],
        temperature=1,
        top_p=1,
        max_tokens=2048,
        use_responses_api=False,
        extra_body={"chat_template_kwargs": policy.rendering_options},
    )
    middleware: list[AgentMiddleware[Any, None]] = [
        ModelCallLimitMiddleware(run_limit=max_turns, exit_behavior="error"),
        _canonical_tools,
    ]
    agent = create_deep_agent(
        model,
        backend=DockerBackend(sandbox),
        middleware=middleware,
        subagents=[],
        system_prompt="Work in /workspace. Solve the Python exercise in INSTRUCTIONS.md. Preserve the required interfaces. Check your solution, then finish.",
    )
    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Solve the exercise in /workspace."}]},
            config={"recursion_limit": max_turns * 6 + 30},
        )
    except ModelCallLimitExceededError:
        return AgentResult(None, "max_turns")
    return AgentResult(result["messages"][-1].text, "completed")


@wrap_model_call
async def _canonical_tools(
    request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
) -> ModelResponse:
    """Bind stable tool schemas before the model SDK records and serves the request."""
    tools = canonical_tool_schemas([convert_to_openai_tool(tool) for tool in request.tools])
    return await handler(request.override(tools=[*tools]))
