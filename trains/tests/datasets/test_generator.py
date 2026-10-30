import pytest

from dataset_fixtures import datum, read_datum, shards_factory
from datasets import Dataset, IterableDataset, load_from_disk
from trains.datasets import (
    chat_dataset_from_records,
    datum_dataset_from_chat_dataset,
    datum_dataset_from_datums,
    datum_dataset_from_generator,
)
from trains.datum.adapters.tinker import TinkerDatumAdapter


def test_streaming_factory_is_lazy_replayable_and_uses_native_batching():
    calls = []

    def generate():
        calls.append("open")
        yield from (datum(i) for i in range(5))

    dataset = datum_dataset_from_generator(generate, streaming=True)
    assert isinstance(dataset, IterableDataset) and calls == []
    assert [len(batch["datum_json"]) for batch in dataset.iter(batch_size=2)] == [2, 2, 1]
    assert len(list(dataset)) == 5 and calls == ["open", "open"]


def test_hf_shards_shuffle_and_repeat_preserve_native_datums():
    dataset = datum_dataset_from_generator(
        shards_factory, gen_kwargs={"shards": [0, 1, 2, 3]}, streaming=True
    )
    assert dataset.num_shards == 4
    shuffled = dataset.shuffle(seed=42, buffer_size=2)
    assert list(shuffled) == list(shuffled)
    assert {read_datum(row["datum_json"]).model_input.to_ints()[0] for row in shuffled} == {
        0,
        1,
        2,
        3,
    }
    assert len(list(dataset.shard(num_shards=2, index=0))) == 2


def test_materialized_generator_writes_native_arrow_and_reloads(tmp_path):
    dataset = datum_dataset_from_generator(
        shards_factory, gen_kwargs={"shards": [1, 2]}, cache_dir=str(tmp_path / "cache")
    )
    assert isinstance(dataset, Dataset) and len(dataset) == 2
    dataset.save_to_disk(tmp_path / "saved")
    assert load_from_disk(tmp_path / "saved")[:] == dataset[:]


def test_empty_generator_and_empty_conversion_have_stable_columns(tmp_path):
    dataset = datum_dataset_from_generator(lambda: iter(()), cache_dir=str(tmp_path))
    converted = datum_dataset_from_chat_dataset(
        chat_dataset_from_records([]), None, TinkerDatumAdapter()
    )
    assert len(dataset) == len(converted) == 0
    assert dataset.features == converted.features


def test_one_shot_iterators_materialize_once_and_streaming_requires_factory():
    source = (datum(i) for i in range(2))
    assert len(datum_dataset_from_datums(source)) == 2
    assert list(source) == []
    with pytest.raises(TypeError, match="factory"):
        datum_dataset_from_generator(iter([datum()]), streaming=True)
