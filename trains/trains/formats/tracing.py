import json
from collections.abc import Callable, Iterator
from typing import Any, Self, cast

from pydantic import BaseModel, Field

from trains.formats.sampling import sampling_parameters
from trains.types import ChatRecord


class TraceRecord(BaseModel):
    """Native spans grouped by trace identity with optional experiment metadata."""

    id: str
    spans: list[dict]
    metadata: dict = Field(default_factory=dict)


class TraceSpec(BaseModel):
    """Select exact native model calls and map client-owned provenance explicitly."""

    format_id: str = "auto"

    def iter_chats(
        self,
        trace: TraceRecord,
        *,
        reward: float | None = None,
        group_id: str | None = None,
        model_id: str | None = None,
    ) -> Iterator[ChatRecord]:
        # Providers re-export these specs through trains.tracing during package import.
        """Yield complete model calls with trace identities and selected rewards."""
        from trains.tracing.adapters.registry import registry  # noqa: PLC0415

        generations, matches = registry.convert(trace.spans, self.format_id)
        ids = [item["span_id"] for item in matches for _ in range(item["generations"])]
        spans = {span["SpanId"]: span["SpanAttributes"] for span in trace.spans}
        for generation, span_id in zip(generations, ids, strict=True):
            record = cast(ChatRecord, dict(generation))
            invocation = spans[span_id].get("llm.invocation_parameters") or {}
            if isinstance(invocation, str):
                invocation = json.loads(invocation)
            metadata = {**trace.metadata, "sampling": sampling_parameters(invocation)}
            record.update(
                {
                    "trace_id": trace.id,
                    "span_id": span_id,
                    "reward": reward if reward is not None else trace.metadata.get("reward"),
                    "group_id": group_id
                    if group_id is not None
                    else trace.metadata.get("example_id"),
                    "model_id": model_id
                    if model_id is not None
                    else spans[span_id].get("llm.model_name"),
                    "metadata_json": json.dumps(metadata, allow_nan=False),
                }
            )
            yield record


class OpenInferenceTraceSpec(TraceSpec):
    """Select LLM calls within a trace or under explicit OpenInference anchor spans.

    AGENT selects descendant LLM calls, never the agent's summarized output.
    Missing ancestry fails rather than training an unrelated model call.
    """

    span_kind: str = "LLM"

    @classmethod
    def from_span_kind(cls, kind: Any, **kwargs: Any) -> Self:
        """Select model calls beneath an OpenInference span kind."""
        return cls(span_kind=getattr(kind, "value", kind), **kwargs)

    def iter_chats(self, trace: TraceRecord, **kwargs: Any) -> Iterator[ChatRecord]:
        """Yield complete model calls with trace identities and selected rewards."""
        anchors = {
            s.get("SpanId")
            for s in trace.spans
            if s.get("SpanAttributes", {}).get("openinference.span.kind") == self.span_kind
        }
        if not anchors:
            return
        parents = {s.get("SpanId"): s.get("ParentSpanId") for s in trace.spans}
        selected = []
        for span in trace.spans:
            if span.get("SpanAttributes", {}).get("openinference.span.kind") != "LLM":
                continue
            current, visited = span.get("SpanId"), set()
            while current and current not in visited and current not in anchors:
                visited.add(current)
                current = parents.get(current)
            if current in anchors:
                selected.append(span)
        if not selected:
            raise ValueError("Selected anchors have no replayable descendant LLM spans")
        yield from super().iter_chats(trace.model_copy(update={"spans": selected}), **kwargs)


Reward = Callable[[TraceRecord], float | None]
Identity = Callable[[TraceRecord], str | None]
