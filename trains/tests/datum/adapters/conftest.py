import json

import pytest

from dataset_fixtures import Processor
from datasets import load_from_disk
from trains.datasets import chat_dataset_from_records, datum_dataset_from_chat_dataset
from trains.datasets.serialization import datum_mapping


@pytest.fixture
def native_roundtrip(tmp_path):

    def check(adapter, datum_type):
        chats = chat_dataset_from_records(
            [
                {
                    "messages": [
                        {"role": "user", "content": "ab"},
                        {"role": "assistant", "content": "cd"},
                    ],
                    "trace_id": "trace",
                    "span_id": "call",
                    "group_id": "task",
                    "model_id": "model",
                    "reward": 0.75,
                    "metadata_json": '{"verifier":"rule"}',
                }
            ]
        )
        chats.save_to_disk(tmp_path / "chats")
        snapshot = load_from_disk(tmp_path / "chats")
        native = adapter.from_tokens([97, 98], [99, 100])
        assert isinstance(native, datum_type)
        result = datum_dataset_from_chat_dataset(snapshot, Processor(), adapter)
        value = json.loads(result[0]["datum_json"])
        assert value == datum_mapping(native)
        assert value["model_input"]["chunks"][0]["tokens"] == [97, 98, 99]
        assert value["loss_fn_inputs"]["target_tokens"]["data"] == [98, 99, 100]
        assert value["loss_fn_inputs"]["weights"]["data"] == [0, 1, 1]
        assert "logprobs" not in value["loss_fn_inputs"]
        for key in ["trace_id", "span_id", "group_id", "model_id", "reward", "metadata_json"]:
            assert result[0][key] == snapshot[0][key]
        alternate = datum_dataset_from_chat_dataset(snapshot, Processor(offset=100), adapter)
        assert json.loads(alternate[0]["datum_json"])["model_input"]["chunks"][0]["tokens"] == [
            197,
            198,
            199,
        ]
        assert snapshot[:] == chats[:]

    return check
