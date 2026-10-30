import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack

import anyio
import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from train.api.dependencies import training
from train.service import TrainingService
from train.serving import Serving, async_locked

router = APIRouter(tags=["Published model inference"])


class InferenceStream(StreamingResponse):
    """A streaming response that retains the shared-GPU lock until closure."""

    def __init__(self, upstream: httpx.Response, stack: AsyncExitStack, model: str) -> None:
        """Retain the upstream stream and serving lock until the response closes.

        Args:
            upstream: Open streaming response from the inference backend.
            stack: Resource stack retaining the serving lock and upstream connection.
            model: Public model identity written into streamed responses.
        """
        self.stack = stack
        super().__init__(
            _relay(upstream, model),
            status_code=upstream.status_code,
            headers={"content-type": upstream.headers.get("content-type", "text/event-stream")},
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Stream the inference response and release its backend resources."""
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.stack.aclose()


@router.post("/v1/chat/completions")
@router.post("/oai/api/v1/chat/completions")
@router.post("/inference/v1/chat/completions")
async def chat(
    body: dict, request: Request, service: TrainingService = Depends(training)
) -> Response:
    """Proxy Chat Completions for the requested published model."""
    return await proxy(body, "chat/completions", service)


@router.post("/v1/completions")
@router.post("/oai/api/v1/completions")
async def completions(body: dict, service: TrainingService = Depends(training)) -> Response:
    """Proxy text completions for the requested published model."""
    return await proxy(body, "completions", service)


@router.post("/v1/responses")
async def responses(body: dict, service: TrainingService = Depends(training)) -> Response:
    """Proxy Responses API requests for the requested published model."""
    return await proxy(body, "responses", service)


@router.post("/v1/messages")
async def messages(body: dict, service: TrainingService = Depends(training)) -> Response:
    """Proxy Anthropic Messages requests to the configured backend."""
    return await proxy(body, "messages", service)


@router.post("/v1/messages/count_tokens")
async def count_tokens(body: dict, service: TrainingService = Depends(training)) -> Response:
    """Proxy native Anthropic token-count requests."""
    return await proxy(body, "messages/count_tokens", service)


@router.post("/deployment/{deployment_id}/sync/v1/chat/completions")
async def baseten_chat(
    deployment_id: str, body: dict, service: TrainingService = Depends(training)
) -> Response:
    """Proxy chat through a native Baseten deployment URL."""
    return await proxy(body, "chat/completions", service, deployment_id)


@router.post("/deployment/{deployment_id}/sync/v1/responses")
async def baseten_responses(
    deployment_id: str, body: dict, service: TrainingService = Depends(training)
) -> Response:
    """Proxy Responses through the checkpoint pinned to a Baseten deployment."""
    return await proxy(body, "responses", service, deployment_id)


async def proxy(
    body: dict, path: str, service: TrainingService, deployment_id: str | None = None
) -> Response:
    """Hold the shared GPU lock until the upstream stream is closed, including disconnects."""
    manager, serving = service.deployments, service.runtime.serving
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("A model identifier is required")
    if "#" in model:
        model, selected_deployment = model.split("#", 1)
        if deployment_id is not None and selected_deployment != deployment_id:
            raise ValueError("Model deployment does not match the request deployment")
        deployment_id = selected_deployment
    stack = AsyncExitStack()
    try:
        await stack.enter_async_context(async_locked(service.root))
        uri, base = manager.resolve(model, deployment_id)
        preparation = asyncio.create_task(asyncio.to_thread(_prepare, service, uri, base))
        try:
            upstream_model = await asyncio.shield(preparation)
        except asyncio.CancelledError:
            with anyio.CancelScope(shield=True):
                await preparation
            raise
        client = await stack.enter_async_context(
            httpx.AsyncClient(headers=serving.headers, timeout=serving.timeout)
        )
        request_body = {
            **_bounded_output(body, path, serving.max_output_tokens),
            "model": upstream_model,
        }
        request_body = await _fit_context(client, request_body, path, serving)
        upstream = await stack.enter_async_context(
            client.stream(
                "POST",
                serving.endpoint("/v1/" + path),
                json=request_body,
            )
        )
        if upstream.status_code >= 400 or not body.get("stream"):
            content = await upstream.aread()
            if upstream.status_code < 400 and "application/json" in upstream.headers.get(
                "content-type", ""
            ):
                content = _rewrite_model(content, body["model"])
            return Response(
                content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type"),
            )
        return InferenceStream(upstream, stack.pop_all(), body["model"])
    finally:
        with anyio.CancelScope(shield=True):
            await stack.aclose()


def _bounded_output(body: dict, path: str, limit: int | None) -> dict:
    """Apply an explicit serving output ceiling without modifying model input."""
    if limit is None or path not in {"responses", "messages", "chat/completions", "completions"}:
        return body
    field = "max_output_tokens" if path == "responses" else "max_tokens"
    fields = [field]
    if path == "chat/completions" and "max_completion_tokens" in body:
        fields = ["max_completion_tokens"] + ([field] if field in body else [])
    bounded = dict(body)
    for field in fields:
        requested = body.get(field)
        if requested is None:
            bounded[field] = limit
        elif isinstance(requested, int) and not isinstance(requested, bool) and requested > 0:
            bounded[field] = min(requested, limit)
    return bounded


async def _fit_context(client: httpx.AsyncClient, body: dict, path: str, serving: Serving) -> dict:
    """Fit text-chat output to native context capacity without shortening the input."""
    messages = body.get("messages")
    if (
        serving.backend != "sglang"
        or serving.max_output_tokens is None
        or path != "chat/completions"
        or not isinstance(messages, list)
        or any(
            not isinstance(message, dict)
            or (message.get("content") is not None and not isinstance(message["content"], str))
            for message in messages
        )
    ):
        return body
    fields = [key for key in ("max_tokens", "max_completion_tokens") if key in body]
    if not fields or any(type(body[key]) is not int or body[key] <= 0 for key in fields):
        return body
    response = await client.post(
        serving.endpoint("/tokenize"),
        json={
            key: value
            for key, value in body.items()
            if key not in {*fields, "stream", "stream_options"}
        },
    )
    if response.is_error:
        return body  # Preserve normal backend validation errors on the actual chat request.
    tokenization = response.json()
    count, context = tokenization.get("count"), tokenization.get("max_model_len")
    if type(count) is not int or type(context) is not int or count < 0 or context <= count:
        return body
    return {**body, **{key: min(body[key], context - count) for key in fields}}


def _prepare(service: TrainingService, uri: str | None, base: str) -> str:
    """Resolve the published model and acquire serving coordination."""
    manager, serving = service.deployments, service.runtime.serving
    with httpx.Client(headers=serving.headers, timeout=serving.timeout) as client:
        manager.check_backend(client, base)
        return manager.ensure_loaded(client, uri, base)


def _rewrite_model(content: bytes, model: str) -> bytes:
    """Rewrite served model IDs in a response to the public checkpoint identity."""
    try:
        value = json.loads(content)
    except ValueError:
        return content
    if isinstance(value, dict):
        if "model" in value:
            value["model"] = model
        for field in ("response", "message"):
            if isinstance(value.get(field), dict) and "model" in value[field]:
                value[field]["model"] = model
        return json.dumps(value).encode()
    return content


async def _relay(upstream: httpx.Response, model: str) -> AsyncIterator[bytes]:
    """Forward an inference request while preserving streaming and error responses."""
    async for line in upstream.aiter_lines():
        content = line.encode()
        if content.startswith(b"data: "):
            content = b"data: " + _rewrite_model(content[6:], model)
        yield content + b"\n"
