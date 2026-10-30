# Select exact SDK model-call spans when harness instrumentation also emits summaries.

from collections.abc import Iterator
from typing import Any

from pydantic import Field
from recipes.tracing import has_completed_output
from trains.formats.tracing import TraceRecord, TraceSpec
from trains.types import ChatRecord

MODEL_CALL_SPANS = {
    "Response",
    "ChatCompletion",
    "messages.create",
    "messages.stream",
    "messages.parse",
    "beta.messages.create",
    "beta.messages.stream",
    "beta.messages.parse",
}


class ModelCallSpec(TraceSpec):
    """Select replayable SDK model calls and enforce receipt rendering settings.

    Examples:
        Export one exact model response while keeping a failed attempt's reward:

        >>> attributes = {
        ...     "openinference.span.kind": "LLM",
        ...     "input.value": {"messages": [{"role": "user", "content": "Solve it"}]},
        ...     "output.value": {"role": "assistant", "content": "Attempt"},
        ... }
        >>> source = TraceRecord(id="attempt-1", metadata={"reward": 0}, spans=[{
        ...     "SpanId": "call-1", "SpanName": "ChatCompletion",
        ...     "SpanAttributes": attributes,
        ... }])
        >>> row, = ModelCallSpec().iter_chats(source)
        >>> row["reward"], row["trace_id"], row["messages"][-1]["content"]
        (0, 'attempt-1', 'Attempt')
    """

    rendering_options: dict = Field(default_factory=dict)

    def iter_chats(self, trace: TraceRecord, **kwargs: Any) -> Iterator[ChatRecord]:
        """Yield complete model calls with trace identities and selected rewards."""
        selected = []
        for span in trace.spans:
            if span.get("SpanName") not in MODEL_CALL_SPANS:
                continue
            if span.get("status_code") == "ERROR":
                attrs = span.get("SpanAttributes") or {}
                if has_completed_output(attrs):
                    selected.append(span)
                    continue
                if attrs.get("output.value") or any(
                    key.startswith("llm.output_messages.") for key in attrs
                ):
                    raise ValueError("Cannot replay an errored request with partial model output")
                continue  # Failed request without a generated action; preserve its successful retry.
            selected.append(span)
        if not selected:
            raise ValueError(
                "No exact model-call spans were exported by the upstream SDK instrumentor"
            )
        for record in super().iter_chats(trace.model_copy(update={"spans": selected}), **kwargs):
            captured = record.get("chat_template_kwargs") or {}
            if captured and captured != self.rendering_options:
                raise ValueError("Captured rendering options differ from the collection receipt")
            yield {**record, "chat_template_kwargs": self.rendering_options}
