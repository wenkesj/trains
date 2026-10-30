import re

from trains.formats.records import HFRecord, validate_record
from trains.tracing.adapters.chat import ChatAdapter
from trains.tracing.adapters.protocol import TraceAdapter
from trains.tracing.adapters.span import Span
from trains.tracing.adapters.types import AdapterMatch


class Registry:
    """Select converters explicitly or detect OpenInference model spans."""

    def __init__(self) -> None:
        """Create an empty registry of native trace-format adapters."""
        self.adapters: dict[str, TraceAdapter] = {}

    def register(self, adapter: TraceAdapter) -> None:
        """Register an adapter and reject duplicate format identities."""
        if adapter.id in self.adapters:
            raise ValueError("Adapter already registered")
        if not re.fullmatch("[a-z0-9][a-z0-9_-]*/v[1-9][0-9]*", adapter.id):
            raise ValueError("Adapters must have versioned IDs")
        self.adapters[adapter.id] = adapter

    def get(self, name: str) -> TraceAdapter:
        """Resolve an explicitly selected adapter by its format identity."""
        if name not in self.adapters:
            raise ValueError(f"Unknown trace format: {name}")
        return self.adapters[name]

    def convert(
        self, rows: list[dict], format_id: str = "auto"
    ) -> tuple[list[HFRecord], list[AdapterMatch]]:
        """Convert spans with matching adapters and report their source identities."""
        result: list[HFRecord] = []
        matched: list[AdapterMatch] = []
        candidates = [self.get(format_id)] if format_id != "auto" else self.adapters.values()
        for row in rows:
            span = Span.from_row(row)
            adapter = next((a for a in candidates if a.matches(span)), None)
            if adapter is None:
                continue
            try:
                values = [validate_record(value) for value in adapter.extract(span)]
                if not values:
                    raise ValueError("No assistant generations found")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Span {span.span_id} [{adapter.id}]: {exc}") from exc
            result.extend(values)
            matched.append(
                {"span_id": span.span_id, "format_id": adapter.id, "generations": len(values)}
            )
        if not result:
            raise ValueError("No supported, replayable model generations found")
        return (result, matched)


def _builtin_registry() -> Registry:
    """Create a registry containing the supported OpenInference format adapters."""
    result = Registry()
    result.register(ChatAdapter())
    return result


registry = _builtin_registry()
