# Expose the frozen Loops token sampler to Pi's OpenAI-compatible client.

import json
import math
import threading
from contextlib import AbstractContextManager
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

from baseten import loops
from tinker_cookbook.renderers import ToolCall

from rendering import LiquidRenderer


class SamplerBridge(AbstractContextManager):
    """Keep exact token receipts while Pi executes its normal coding-agent loop."""

    def __init__(self, sampler, renderer, checkpoint, *, max_tokens=2048, max_calls=20):
        """Bind the frozen sampler, native renderer, and per-attempt call budget."""
        self.sampler, self.renderer, self.checkpoint = sampler, renderer, checkpoint
        self.max_tokens, self.max_calls = max_tokens, max_calls
        self.calls, self.errors = [], []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_RequestHandler, self))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def complete(self, body):
        """Sample from the frozen policy and retain exact native token receipts."""
        if len(self.calls) >= self.max_calls:
            raise ValueError("Coding-agent model-call budget exhausted")
        if body.get("model") != "policy":
            raise ValueError("Select the published policy")
        prompt_ids = self._prompt(body)
        sample, logprobs = self._sample(prompt_ids)
        output, termination = self.renderer.parse_response(sample.tokens)
        message = wire_message(output)
        # XML calls have no IDs; allocate unique IDs before Pi persists them.
        for call in message.get("tool_calls") or []:
            call["id"] = "call_" + uuid4().hex
        self.calls.append(
            {
                "messages": body["messages"],
                "tools": body.get("tools") or [],
                "output": message,
                "prompt_tokens": prompt_ids,
                "completion_tokens": sample.tokens,
                "sampling_logprobs": logprobs,
                "stop_reason": sample.stop_reason,
                "clean": termination.is_clean,
                "checkpoint": self.checkpoint,
            }
        )
        return _completion_response(message)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.thread.join(timeout=10)
        self.server.server_close()

    def _prompt(self, body):
        """Render the original wire conversation with the configured native template."""
        if isinstance(self.renderer, LiquidRenderer):
            prompt = self.renderer.build_generation_prompt(
                body["messages"],
                tools=body.get("tools") or [],
            )
        else:
            native = native_messages(body["messages"], body.get("tools") or [], self.renderer)
            prompt = self.renderer.build_generation_prompt(native)
        return [token for chunk in prompt.chunks for token in chunk.tokens]

    def _sample(self, prompt_ids):
        """Validate native action probabilities at untruncated temperature one."""
        sample = self.sampler.sample(
            loops.ModelInput(chunks=[loops.EncodedTextChunk(tokens=prompt_ids)]),
            sampling_params=loops.SamplingParams(
                max_tokens=self.max_tokens,
                temperature=1,
                top_p=1,
                top_k=-1,
                stop=self.renderer.get_stop_sequences(),
            ),
        ).sequences[0]
        logprobs = getattr(sample, "logprobs", None)
        if logprobs is not None and (
            len(logprobs) != len(sample.tokens)
            or not all(math.isfinite(value) for value in logprobs)
        ):
            raise ValueError("Native sampler returned misaligned or nonfinite logprobs")
        return sample, logprobs


class _RequestHandler(BaseHTTPRequestHandler):
    """Translate Pi HTTP requests without changing native actions or failures."""

    def __init__(self, bridge, *args, **kwargs):
        """Bind the per-attempt bridge before handling the request."""
        self.bridge = bridge
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: object) -> None:
        """Keep HTTP access messages out of native session output."""

    def do_POST(self):
        """Handle native completions while recording all request failures."""
        try:
            if self.path != "/v1/chat/completions":
                raise ValueError("Only chat completions are supported")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            response = self.bridge.complete(body)
            self.send_response(200)
            if body.get("stream"):
                self._stream(response)
            else:
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
        except Exception as error:  # noqa: BLE001 - HTTP boundary retains sampler failures.
            self.bridge.errors.append(str(error))
            self.send_error(400, str(error))

    def _stream(self, response):
        """Send the complete sampled action and terminal event in native SSE form."""
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for delta, reason in [
            (response["choices"][0]["message"], None),
            ({}, response["choices"][0]["finish_reason"]),
        ]:
            chunk = {
                "id": response["id"],
                "object": "chat.completion.chunk",
                "model": "policy",
                "created": 0,
                "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
            }
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")


def native_messages(messages, tools, renderer):
    """Translate wire messages and tool schemas into the native training renderer."""
    values = []
    for message in messages:
        value = {
            key: item
            for key, item in message.items()
            if key in {"role", "content", "tool_call_id", "name"}
        }
        content = value.get("content") or ""
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise ValueError("This coding example accepts text-only requests")
            content = "".join(part["text"] for part in content)
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        value["content"] = (
            [
                {"type": "thinking", "thinking": reasoning},
                {"type": "text", "text": content},
            ]
            if reasoning
            else content
        )
        if message.get("tool_calls"):
            value["tool_calls"] = [ToolCall.model_validate(call) for call in message["tool_calls"]]
        values.append(value)
    if tools:
        system = ""
        if values and values[0]["role"] == "system":
            system = values.pop(0)["content"]
        schemas = [
            {
                **tool["function"],
                "description": tool["function"].get("description", ""),
                "parameters": tool["function"].get("parameters", {}),
            }
            for tool in tools
        ]
        values = renderer.create_conversation_prefix_with_tools(schemas, system) + values
    return values


def wire_message(native):
    """Return the assistant output Pi will persist in its own session file."""
    parts = native.get("content") or ""
    result = {"role": "assistant", "content": parts if isinstance(parts, str) else ""}
    if isinstance(parts, list):
        result["content"] = "".join(p["text"] for p in parts if p["type"] == "text")
        thinking = "".join(p["thinking"] for p in parts if p["type"] == "thinking")
        if thinking:
            result["reasoning_content"] = thinking
    if native.get("tool_calls"):
        result["tool_calls"] = [call.model_dump(mode="json") for call in native["tool_calls"]]
    return result


def _completion_response(message):
    """Add transport metadata while keeping persisted tool-call IDs stable."""
    delta = dict(message)
    if delta.get("tool_calls"):
        delta["tool_calls"] = [
            {**call, "index": index} for index, call in enumerate(delta["tool_calls"])
        ]
    return {
        "id": "chatcmpl-" + uuid4().hex,
        "object": "chat.completion",
        "created": 0,
        "model": "policy",
        "choices": [
            {
                "index": 0,
                "message": delta,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            }
        ],
    }
