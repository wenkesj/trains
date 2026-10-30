import asyncio

import httpx2 as httpx
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from recipes.tracing import ModelCallRecorder

from claude_tracing.spec import MODEL_CALL_SPANS


def test_actual_async_sdk_call_is_recorded_under_attempt_trace():
    provider = TracerProvider()
    recorder = ModelCallRecorder(MODEL_CALL_SPANS)
    provider.add_span_processor(recorder)
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=provider)

    async def generate():
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "id": "msg_local",
                    "type": "message",
                    "role": "assistant",
                    "model": "local",
                    "content": [{"type": "text", "text": "hello"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                },
            )
        )
        async with AsyncAnthropic(
            api_key="local-only", http_client=httpx.AsyncClient(transport=transport)
        ) as client:
            with provider.get_tracer(__name__).start_as_current_span("attempt") as root:
                result = await client.messages.create(
                    model="local", max_tokens=8, messages=[{"role": "user", "content": "hello"}]
                )
                assert isinstance(result.content[0], TextBlock)
                assert result.content[0].text == "hello"
                trace_id = f"{root.get_span_context().trace_id:032x}"
        return trace_id

    try:
        trace_id = asyncio.run(generate())
        calls = recorder.take(trace_id)
        assert len(calls) == 1 and len(calls[0]) == 16
        assert recorder.take(trace_id) == []
    finally:
        instrumentor.uninstrument()
        provider.shutdown()
