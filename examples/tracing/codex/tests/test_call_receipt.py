import asyncio

import httpx
from openai import AsyncOpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from recipes.tracing import ModelCallRecorder

from codex_tracing.spec import MODEL_CALL_SPANS


def test_actual_async_sdk_call_is_recorded_under_attempt_trace():
    provider = TracerProvider()
    recorder = ModelCallRecorder(MODEL_CALL_SPANS)
    provider.add_span_processor(recorder)
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(tracer_provider=provider)

    async def generate():
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "id": "chatcmpl_local",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "local",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                },
            )
        )
        async with AsyncOpenAI(
            api_key="local-only", http_client=httpx.AsyncClient(transport=transport)
        ) as client:
            with provider.get_tracer(__name__).start_as_current_span("attempt") as root:
                result = await client.chat.completions.create(
                    model="local", messages=[{"role": "user", "content": "hello"}]
                )
                assert result.choices[0].message.content == "hello"
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
