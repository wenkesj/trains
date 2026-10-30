import pytest

from dataset_fixtures import Processor, read_datum
from datasets import Dataset, IterableDataset
from trains.datasets import (
    chat_dataset_from_records,
    datum_dataset_from_chat_dataset,
    datum_dataset_from_datums,
)
from trains.datum.adapters.tinker import TinkerDatumAdapter
from trains.formats.tokenization import encode


def test_empty_dataset_has_stable_schema_and_no_fake_rows():
    dataset = datum_dataset_from_datums([])
    assert len(dataset) == 0 and dataset.features["reward"].dtype == "float64"


def test_generation_masks_prompt_without_inventing_policy_scores(processor):
    chats = chat_dataset_from_records(
        [
            {
                "messages": [
                    {"role": "user", "content": "go"},
                    {"role": "assistant", "content": "north"},
                ]
            }
        ]
    )
    dataset = datum_dataset_from_chat_dataset(chats, processor, TinkerDatumAdapter())
    datum = read_datum(dataset[0]["datum_json"])
    assert datum.loss_fn_inputs["weights"].data == [0, 0, 1]
    assert "advantages" not in datum.loss_fn_inputs and "logprobs" not in datum.loss_fn_inputs
    assert datum.model_input.to_ints() == [1, 2, 3]


@pytest.mark.parametrize("record", [{"text": "go north"}, {"prompt": "go", "completion": "north"}])
def test_legacy_records_are_rejected(processor, record):
    with pytest.raises(ValueError, match="messages"):
        chat_dataset_from_records([record])
    with pytest.raises(ValueError, match="messages"):
        encode(record, processor)


def test_native_hf_records_can_be_converted_without_trains_export():
    dataset = Dataset.from_list(
        [{"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}]
    )
    result = datum_dataset_from_chat_dataset(dataset, Processor(), TinkerDatumAdapter())
    assert len(result) == 1 and result[0]["trace_id"] is None


def test_empty_stream_and_multiprocess_conversion():
    source = chat_dataset_from_records(
        [{"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}]
        * 4
    )
    result = datum_dataset_from_chat_dataset(source, Processor(), TinkerDatumAdapter(), num_proc=2)
    assert len(result) == 4
    empty = datum_dataset_from_chat_dataset(
        IterableDataset.from_generator(
            lambda: iter(()), features=chat_dataset_from_records([]).features
        ),
        Processor(),
        TinkerDatumAdapter(),
    )
    assert list(empty) == [] and empty.features == result.features


def test_hub_messages_with_prompt_metadata_are_accepted(processor):
    source = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "go"},
                    {"role": "assistant", "content": "north"},
                ],
                "prompt": "metadata, not the tokenized input",
                "source": "hub",
            }
        ]
    )
    data = datum_dataset_from_chat_dataset(source, processor, TinkerDatumAdapter())
    assert read_datum(data[0]["datum_json"]).model_input.to_ints() == [1, 2, 3]
