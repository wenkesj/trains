# Route native model protocols through their real SDKs and upstream instrumentors.
#
# No agent transcript or model response is reconstructed here. The same request is
# sent through the native SDK to the published local policy. This supplies model
# boundary instrumentation when a subprocess harness only exposes agent summaries.

import copy
import inspect
import os
import socket
import threading
import time
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, Self

import anthropic
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import context
from recipes.tracing import canonical_tool_schemas

from claude_tracing.training import Policy


class ModelProxy(AbstractContextManager):
    """Route harness requests to the frozen policy through an instrumented SDK."""

    def __init__(self, policy: Policy) -> None:
        """Route harness calls through the frozen policy and its native protocol.

        Args:
            policy: Frozen collection policy and its model endpoint.
        """
        self.policy = policy
        self.parent = context.get_current()
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.socket.getsockname()[1]}"
        self.server = uvicorn.Server(
            uvicorn.Config(self._application(), log_level="warning", access_log=False)
        )
        self.worker = threading.Thread(
            target=self.server.run, kwargs={"sockets": [self.socket]}, daemon=True
        )

    def _application(self) -> FastAPI:
        """Build the temporary harness-facing model API."""
        app = FastAPI()

        @app.get("/v1/models")
        async def models() -> dict:
            return {"object": "list", "data": [{"id": self.policy.model, "object": "model"}]}

        @app.post("/v1/{path:path}")
        async def forward(path: str, request: Request) -> Any:
            if path not in {"messages", "messages/count_tokens"}:
                raise HTTPException(status_code=404)
            try:
                body = _canonical_request(await request.json())
            except (TypeError, ValueError) as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            if body.get("model") not in {self.policy.model, "policy"}:
                raise HTTPException(status_code=422, detail="Select the published behavior policy")
            body["model"] = self.policy.model
            if path != "messages/count_tokens":
                for name in ("temperature", "top_p"):
                    if body.get(name, 1) != 1:
                        raise HTTPException(
                            status_code=422,
                            detail="This recipe requires untruncated temperature=1 sampling",
                        )
                    body[name] = 1
                body["chat_template_kwargs"] = self.policy.rendering_options
            headers = {
                name: request.headers[name]
                for name in ("anthropic-version", "anthropic-beta")
                if name in request.headers
            }
            if body.get("stream"):
                return StreamingResponse(
                    self._stream(path, body, headers), media_type="text/event-stream"
                )
            token = context.attach(self.parent)
            try:
                async with self._client(path) as client:
                    response = await _invoke(client, path, body, headers)
                    return JSONResponse(response.model_dump(mode="json"))
            except anthropic.APIStatusError as error:
                return JSONResponse({"error": str(error)}, status_code=error.status_code)
            finally:
                context.detach(token)

        return app

    def _client(self, path: str) -> Any:
        """Return the native SDK client for this request protocol."""
        key = os.environ["TINKER_API_KEY"]
        return anthropic.AsyncAnthropic(
            base_url=self.policy.inference_url.removesuffix("/v1"), api_key=key, max_retries=0
        )

    async def _stream(self, path: str, body: dict, headers: dict) -> Any:
        """Stream native model events without altering the captured model call."""
        token = context.attach(self.parent)
        try:
            async with self._client(path) as client:
                stream = await _invoke(client, path, body, headers)
                async with stream:
                    async for event in stream:
                        payload = event.model_dump_json()
                        yield f"event: {event.type}\ndata: {payload}\n\n".encode()
        finally:
            context.detach(token)

    def __enter__(self) -> Self:
        self.worker.start()
        deadline = time.monotonic() + 30
        while self.worker.is_alive() and not self.server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self.server.started:
            self.__exit__(None, None, None)
            raise RuntimeError("Model protocol routing server did not start")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.server.should_exit = True
        if self.worker.is_alive():
            self.worker.join(timeout=15)
        self.socket.close()
        if self.worker.is_alive():
            raise RuntimeError("Model protocol routing server did not stop")


def _canonical_request(body: dict) -> dict:
    """Make native system/tool defaults explicit before SDK instrumentation."""
    body = copy.deepcopy(body)
    system = _system_parts(body.get("system") or "")
    messages = []
    for message in body.get("messages", []):
        if message.get("role") != "system":
            messages.append(message)
            continue
        system.extend(_system_parts(message.get("content") or ""))
    if system:
        body["system"] = "\n".join(system)
    else:
        body.pop("system", None)
    body["messages"] = messages
    if body.get("tools"):
        tools = []
        for tool in body["tools"]:
            if "input_schema" not in tool:
                tools.append(tool)
                continue
            # Native Function serialization orders these fields and defaults strict
            # to null. Explicit false preserves ordinary non-strict tool semantics
            # while keeping the SDK capture and backend template identical.
            tools.append(
                {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "input_schema": canonical_tool_schemas([tool["input_schema"]])[0],
                    "strict": False if tool.get("strict") is None else tool["strict"],
                    **{
                        key: value
                        for key, value in tool.items()
                        if key not in {"name", "description", "input_schema", "strict"}
                    },
                }
            )
        body["tools"] = tools
    return body


def _system_parts(content: Any) -> list[str]:
    """Retain native text block boundaries and reject unsupported system content."""
    if isinstance(content, str):
        return [content] if content.strip() else []
    if not isinstance(content, list) or any(
        not isinstance(block, dict)
        or block.get("type") != "text"
        or not isinstance(block.get("text"), str)
        for block in content
    ):
        raise ValueError("Anthropic system messages require text or text blocks")
    return [block["text"] for block in content if block["text"]]


def _method(client: Any, path: str) -> Any:
    """Resolve the native SDK method for a model request."""
    return (
        client.messages.count_tokens if path == "messages/count_tokens" else client.messages.create
    )


async def _invoke(client: Any, path: str, body: dict, headers: dict) -> Any:
    """Call the native SDK with the selected policy and rendering settings."""
    call = _method(client, path)
    supported = inspect.signature(call).parameters
    arguments = {key: value for key, value in body.items() if key in supported}
    extra = {key: value for key, value in body.items() if key not in supported}
    return await call(**arguments, extra_body=extra, extra_headers=headers)
