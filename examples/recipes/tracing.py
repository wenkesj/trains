# Record completed SDK calls for native experiment export completeness checks.

import json
from collections import defaultdict
from collections.abc import Mapping, Set
from threading import Lock
from typing import Any
from uuid import UUID

try:
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
    from opentelemetry.trace import StatusCode
except ImportError as error:
    raise ImportError(
        "Tracing recipes require opentelemetry-sdk. Run uv sync in the tracing example project."
    ) from error


class ModelCallRecorder(SpanProcessor):
    """Retain only completed model-call IDs until their attempt receipt is saved."""

    def __init__(self, span_names: Set[str]) -> None:
        """Select the upstream SDK span names that represent actual model calls.

        Args:
            span_names: Exact model-call names, excluding agent summary spans.
        """
        self._names = frozenset(span_names)
        self._calls: dict[str, set[str]] = defaultdict(set)
        self._lock = Lock()

    def on_end(self, span: ReadableSpan) -> None:
        """Record completed SDK generations without retaining their payloads."""
        if span.name not in self._names:
            return
        if span.status.status_code == StatusCode.ERROR and not has_completed_output(
            span.attributes or {}
        ):
            return
        context = span.get_span_context()
        if context is not None:
            with self._lock:
                self._calls[f"{context.trace_id:032x}"].add(f"{context.span_id:016x}")

    def take(self, trace_id: str) -> list[str]:
        """Return completed call IDs and release the attempt's recorder state."""
        with self._lock:
            return sorted(self._calls.pop(trace_id, set()))


def canonical_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order tool-schema keys before inference so provider exports replay them exactly.

    Args:
        tools: Native API tool definitions, in their original list order.

    Returns:
        Independent tool definitions with recursively sorted object keys. Array
        order and schema values remain intact.
    """
    return json.loads(json.dumps(tools, sort_keys=True, allow_nan=False))


def has_completed_output(attributes: Mapping[str, Any]) -> bool:
    """Check native API completion fields when span status also reports an error.

    Args:
        attributes: OpenInference attributes containing the captured API response.

    Returns:
        Whether the response explicitly describes a completed model generation.
    """
    output = attributes.get("output.value")
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError:
            return False
    if not isinstance(output, Mapping) or output.get("error"):
        return False
    if output.get("object") == "response":
        items = output.get("output")
        return (
            output.get("status") == "completed"
            and not output.get("incomplete_details")
            and isinstance(items, list)
            and bool(items)
            and all(
                isinstance(item, Mapping) and item.get("status") in {None, "completed"}
                for item in items
            )
        )
    if output.get("object") == "chat.completion":
        choices = output.get("choices")
        return (
            isinstance(choices, list)
            and len(choices) == 1
            and isinstance(choices[0], Mapping)
            and choices[0].get("finish_reason") in {"stop", "tool_calls", "function_call"}
        )
    return output.get("type") == "message" and output.get("stop_reason") in {
        "end_turn",
        "tool_use",
        "stop_sequence",
    }


def langsmith_run_id(span_id: str) -> str:
    """Convert an OpenTelemetry span ID to LangSmith's native OTLP run UUID.

    Args:
        span_id: The sixteen hexadecimal digits of a completed SDK span.

    Returns:
        The zero-padded native LangSmith run identity.

    Raises:
        ValueError: The receipt does not contain a valid OpenTelemetry span ID.
    """
    if len(span_id) != 16 or any(c not in "0123456789abcdef" for c in span_id):
        raise ValueError("Receipt needs sixteen-digit OpenTelemetry SDK span IDs")
    return str(UUID(int=int(span_id, 16)))
