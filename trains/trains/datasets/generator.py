from collections.abc import Callable, Iterable
from typing import Any
from uuid import uuid4

from datasets import Dataset, Features, IterableDataset, config


def from_generator(
    generator: Callable[..., Iterable[Any]],
    features: Features,
    *,
    gen_kwargs: dict[str, Any] | None = None,
    streaming: bool = False,
    cache_dir: str | None = None,
    fingerprint: str | None = None,
    num_proc: int | None = None,
) -> Dataset | IterableDataset:
    """Preserve native HF factory, cache, and source-shard semantics."""
    if not callable(generator):
        raise TypeError("Supply a replayable generator factory, not an iterator instance")
    if streaming:
        if cache_dir is not None or fingerprint is not None or num_proc is not None:
            raise ValueError("cache_dir, fingerprint and num_proc apply to materialized datasets")
        return IterableDataset.from_generator(generator, features=features, gen_kwargs=gen_kwargs)
    try:
        return Dataset.from_generator(
            generator,
            features=features,
            gen_kwargs=gen_kwargs,
            cache_dir=cache_dir if cache_dir is not None else str(config.HF_DATASETS_CACHE),
            fingerprint=fingerprint or uuid4().hex,
            num_proc=num_proc,
        )
    except ValueError as exc:
        if str(exc) != 'Instruction "train" corresponds to no data!':
            raise
        return Dataset.from_dict({key: [] for key in features}, features=features)
