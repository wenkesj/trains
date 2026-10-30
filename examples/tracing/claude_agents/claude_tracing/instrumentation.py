from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from recipes.tracing import ModelCallRecorder

from claude_tracing.spec import MODEL_CALL_SPANS

model_calls = ModelCallRecorder(MODEL_CALL_SPANS)


def configure(
    *, project: str, endpoint: str | None, headers: dict[str, str] | None = None
) -> TracerProvider:
    """Install upstream OpenInference instrumentors and an optional OTLP exporter."""
    provider = TracerProvider(
        resource=Resource.create({"openinference.project.name": project}),
        span_limits=SpanLimits(max_attributes=10000),
    )
    provider.add_span_processor(model_calls)
    if endpoint is not None:
        provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
        )
    trace.set_tracer_provider(provider)
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=provider)
    AnthropicInstrumentor().instrument(tracer_provider=provider)
    return provider
