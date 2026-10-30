import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import partial
from typing import Any, overload

from datasets import Dataset, IterableDataset
from trains.datasets.generator import from_generator
from trains.datasets.rows import DATUM_FEATURES as FEATURES
from trains.datasets.rows import provenance
from trains.datasets.serialization import datum_mapping
from trains.datum.adapters.protocol import Adapter
from trains.types import DatumRow


def datum_dataset_from_datums(datums: Iterable[Any]) -> Dataset:
    """Serialize finite native SDK datums; native types own datum validation.

    Args:
        datums: Finite iterable of provider-native datums or native datum mappings.

    Returns:
        A materialized HF dataset containing datum_json and empty provenance columns.
    """
    rows = [_datum_row(value, {}) for value in datums]
    return Dataset.from_dict(
        {key: [row[key] for row in rows] for key in FEATURES}, features=FEATURES
    )


def datum_dataset_from_generator(
    generator: Callable[..., Iterable[Any]],
    *,
    gen_kwargs: dict[str, Any] | None = None,
    streaming: bool = False,
    cache_dir: str | None = None,
    fingerprint: str | None = None,
    num_proc: int | None = None,
) -> Dataset | IterableDataset:
    """Serialize native datums with HF generator, caching, and source-shard semantics.

    Args:
        generator: Replayable factory yielding native datums; do not pass an iterator instance.
        gen_kwargs: Arguments passed to the generator; list values can identify source shards.
        streaming: Return a lazy IterableDataset instead of materializing all rows.
        cache_dir: Optional directory for Hugging Face generator caches.
        fingerprint: Explicit Hugging Face dataset fingerprint when deterministic caching is needed.
        num_proc: Number of worker processes used by native HF mapping or generation.

    Returns:
        A materialized or lazy HF dataset of serialized native datums.
    """
    if not callable(generator):
        raise TypeError("Supply a replayable generator factory, not an iterator instance")
    return from_generator(
        partial(_datum_rows, generator),
        FEATURES,
        gen_kwargs=gen_kwargs,
        streaming=streaming,
        cache_dir=cache_dir,
        fingerprint=fingerprint,
        num_proc=num_proc,
    )


@overload
def datum_dataset_from_dataset(
    dataset: Dataset,
    processing_class: Any,
    adapter: Adapter[Mapping[str, Any], Any],
    *,
    num_proc: int | None = None,
) -> Dataset: ...


@overload
def datum_dataset_from_dataset(
    dataset: IterableDataset,
    processing_class: Any,
    adapter: Adapter[Mapping[str, Any], Any],
    *,
    num_proc: int | None = None,
) -> IterableDataset: ...


def datum_dataset_from_dataset(
    dataset: Dataset | IterableDataset,
    processing_class: Any,
    adapter: Adapter[Mapping[str, Any], Any],
    *,
    num_proc: int | None = None,
) -> Dataset | IterableDataset:
    """Render records into zero or more native datums, retaining source provenance.

    Args:
        dataset: HF dataset or iterable of complete model-visible conversation records.
        processing_class: Exact inference tokenizer, media processor, or native renderer.
        adapter: Provider-native datum adapter preserving masks, tools, and media layout.
        num_proc: Number of worker processes used by native HF mapping or generation.

    Returns:
        Provider-native serialized datums with assistant masks and source provenance.
    """
    if isinstance(dataset, IterableDataset) and dataset.column_names is None:
        if num_proc is not None:
            raise ValueError("num_proc applies to materialized datasets")
        return IterableDataset.from_generator(
            partial(_convert_stream, dataset, processing_class, adapter), features=FEATURES
        )
    function = partial(_convert_batch, processing_class=processing_class, adapter=adapter)
    if isinstance(dataset, IterableDataset):
        if num_proc is not None:
            raise ValueError("num_proc applies to materialized datasets")
        return dataset.map(
            function,
            batched=True,
            batch_size=32,
            remove_columns=dataset.column_names,
            features=FEATURES,
        )
    if isinstance(dataset, Dataset) and not len(dataset):
        return Dataset.from_dict({key: [] for key in FEATURES}, features=FEATURES)
    return dataset.map(
        function,
        batched=True,
        batch_size=32,
        remove_columns=dataset.column_names,
        features=FEATURES,
        num_proc=num_proc,
        load_from_cache_file=False,
    )


def _datum_row(datum: Any, record: Mapping[str, Any]) -> DatumRow:
    return {
        **provenance(record),
        "datum_json": json.dumps(datum_mapping(datum), allow_nan=False),
    }


def _datum_rows(generator: Callable[..., Iterable[Any]], **kwargs: Any) -> Iterator[DatumRow]:
    """Yield serialized native datums while retaining conversation provenance."""
    for value in generator(**kwargs):
        yield _datum_row(value, {})


def _convert_batch(
    batch: Mapping[str, list[Any]],
    *,
    processing_class: Any,
    adapter: Adapter[Mapping[str, Any], Any],
) -> dict[str, list[Any]]:
    """Expand a batch of conversations into aligned native datum rows."""
    result = {key: [] for key in FEATURES}
    for index in range(len(next(iter(batch.values())))):
        row = {key: values[index] for key, values in batch.items()}
        for native in adapter.encode(row, processing_class):
            output = _datum_row(native, row)
            for key in FEATURES:
                result[key].append(output[key])
    return result


def _convert_stream(
    dataset: Dataset | IterableDataset,
    processing_class: Any,
    adapter: Adapter[Mapping[str, Any], Any],
) -> Iterator[DatumRow]:
    """Lazily render conversations into native datum rows."""
    for row in dataset:
        for native in adapter.encode(row, processing_class):
            yield _datum_row(native, row)


datum_dataset_from_chat_dataset = datum_dataset_from_dataset
