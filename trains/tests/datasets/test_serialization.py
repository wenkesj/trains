import json

import pytest
from tinker import types

from dataset_fixtures import datum, raw_datum, read_datum
from datasets import Dataset, load_from_disk
from trains.datasets import datum_dataset_from_datums


def test_native_datums_roundtrip_hf_disk_shuffle_batches_and_sdk(tmp_path):
    dataset = datum_dataset_from_datums([raw_datum(), raw_datum(), raw_datum()])
    assert isinstance(dataset, Dataset)
    dataset.save_to_disk(tmp_path / "data")
    restored = load_from_disk(tmp_path / "data")
    assert restored[:] == dataset[:]
    assert [len(batch["datum_json"]) for batch in restored.shuffle(seed=42).iter(batch_size=2)] == [
        2,
        1,
    ]
    value = json.loads(restored[0]["datum_json"])
    datum = types.Datum(
        model_input=types.ModelInput.model_validate(value["model_input"]),
        loss_fn_inputs={
            key: types.TensorData(**tensor) for key, tensor in value["loss_fn_inputs"].items()
        },
    )
    assert datum.model_input.to_ints() == [1, 2]
    assert datum.loss_fn_inputs["target_tokens"].data == [2, 3]


def test_native_datum_schema_is_owned_by_the_sdk():
    with pytest.raises(ValueError):
        types.ModelInput.model_validate({"chunks": [{"type": "invalid"}]})
    assert read_datum(datum_dataset_from_datums([raw_datum()])[0]["datum_json"])


def test_native_tinker_roundtrips_without_a_unified_model():
    before = datum()
    dataset = datum_dataset_from_datums([before])
    restored = read_datum(dataset[0]["datum_json"])
    assert restored.model_input == before.model_input
    assert {k: v.data for k, v in restored.loss_fn_inputs.items()} == {
        k: v.data for k, v in before.loss_fn_inputs.items()
    }
