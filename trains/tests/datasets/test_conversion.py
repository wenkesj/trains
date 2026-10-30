import json

from datasets import Dataset, IterableDataset
from trains.datasets.datums import datum_dataset_from_dataset


class ExpandingAdapter:
    def encode(self, record, processing_class):
        for choice in record["choices"]:
            yield {"choice": choice, "context": record["context"]}


def test_expansion_preserves_provenance_and_streaming():
    source = Dataset.from_list(
        [{"context": "route", "choices": ["a", "b"], "trace_id": "trace", "reward": 1.0}]
    )
    for data in [source, source.to_iterable_dataset()]:
        result = list(datum_dataset_from_dataset(data, None, ExpandingAdapter()))
        assert len(result) == 2
        assert [json.loads(row["datum_json"])["choice"] for row in result] == ["a", "b"]
        assert all(row["trace_id"] == "trace" and row["reward"] == 1.0 for row in result)


def test_untyped_native_hf_stream_expands_without_leaking_input_columns():
    source = IterableDataset.from_generator(
        lambda: iter([{"context": "route", "choices": ["a", "b"], "trace_id": "trace"}])
    )
    output = list(datum_dataset_from_dataset(source, None, ExpandingAdapter()))
    assert len(output) == 2 and all("context" not in row for row in output)
