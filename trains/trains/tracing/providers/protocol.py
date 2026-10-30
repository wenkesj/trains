from collections.abc import AsyncIterator, Iterator
from typing import Protocol

from trains.formats.tracing import Identity, Reward, TraceRecord, TraceSpec


class TraceMetadata(Protocol):
    """Shared format and provenance selectors for tracing providers."""

    @property
    def spec(self) -> TraceSpec:
        """Return the format spec used to decode exported model calls."""
        ...

    @property
    def reward(self) -> Reward | None:
        """Return the selector for the terminal reward attached to each trace."""
        ...

    @property
    def group_id(self) -> Identity | None:
        """Return the selector for grouping repeated attempts of the same task."""
        ...

    @property
    def model_id(self) -> Identity | None:
        """Return the selector for the collected model or sampler identity."""
        ...


class TracingProvider(TraceMetadata, Protocol):
    """A synchronous source of trace records and provenance selectors."""

    def iter_traces(self) -> Iterator[TraceRecord]:
        """Yield native trace exports joined to the selected experiment provenance."""
        ...


class AsyncTracingProvider(TraceMetadata, Protocol):
    """An asynchronous source of trace records and provenance selectors."""

    def aiter_traces(self) -> AsyncIterator[TraceRecord]:
        """Asynchronously yield native traces with selected provenance."""
        ...
