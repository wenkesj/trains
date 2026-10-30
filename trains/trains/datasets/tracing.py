from collections.abc import Callable, Iterator
from functools import partial
from typing import Literal, cast, overload

from datasets import Dataset, IterableDataset
from trains.datasets.chat import FEATURES, chat_dataset_from_records, chat_row
from trains.datasets.generator import from_generator
from trains.tracing.providers.protocol import AsyncTracingProvider, TracingProvider
from trains.types import ChatRow


def chats_from_tracing_provider(
    provider: TracingProvider | Callable[[], TracingProvider],
) -> Iterator[ChatRow]:
    """Export replayable records and client grades without loading a tokenizer or SDK."""
    source = cast(Callable[[], TracingProvider], provider)() if callable(provider) else provider
    seen = set()
    for trace in source.iter_traces():
        if trace.id in seen:
            raise ValueError("Provider returned a duplicate trace")
        seen.add(trace.id)
        for row in source.spec.iter_chats(
            trace,
            reward=source.reward(trace) if source.reward else None,
            group_id=source.group_id(trace) if source.group_id else None,
            model_id=source.model_id(trace) if source.model_id else None,
        ):
            yield chat_row(row)


@overload
def chat_dataset_from_tracing_provider(
    provider: TracingProvider | Callable[[], TracingProvider], *, streaming: Literal[False] = False
) -> Dataset: ...


@overload
def chat_dataset_from_tracing_provider(
    provider: TracingProvider | Callable[[], TracingProvider], *, streaming: Literal[True]
) -> IterableDataset: ...


@overload
def chat_dataset_from_tracing_provider(
    provider: TracingProvider | Callable[[], TracingProvider], *, streaming: bool = False
) -> Dataset | IterableDataset: ...


def chat_dataset_from_tracing_provider(
    provider: TracingProvider | Callable[[], TracingProvider], *, streaming: bool = False
) -> Dataset | IterableDataset:
    """Materialize or stream complete model calls from a native tracing provider.

    Args:
        provider: Initialized native tracing provider or a replayable provider factory.
        streaming: Return a lazy IterableDataset instead of materializing all rows.

    Returns:
        Complete model-call rows with selected trace and experiment provenance.
    """
    if streaming and not callable(provider):
        raise TypeError(
            "Streaming native clients require a provider factory that opens a client per iteration"
        )
    return from_generator(
        partial(chats_from_tracing_provider, provider), FEATURES, streaming=streaming
    )


async def async_chat_dataset_from_tracing_provider(provider: AsyncTracingProvider) -> Dataset:
    """Materialize an async native export; HF iterable factories are synchronous."""
    rows, seen = [], set()
    async for trace in provider.aiter_traces():
        if trace.id in seen:
            raise ValueError("Provider returned a duplicate trace")
        seen.add(trace.id)
        rows.extend(
            provider.spec.iter_chats(
                trace,
                reward=provider.reward(trace) if provider.reward else None,
                group_id=provider.group_id(trace) if provider.group_id else None,
                model_id=provider.model_id(trace) if provider.model_id else None,
            )
        )
    return chat_dataset_from_records(rows)
