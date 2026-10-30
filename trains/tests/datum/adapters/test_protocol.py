import json

from trains.datasets import chat_dataset_from_records, datum_dataset_from_chat_dataset


def test_custom_adapter_can_use_an_unrelated_native_schema():

    class Processor:
        def encode(self, text, **kwargs):
            return [1] if text == "input" else [2]

    class CustomAdapter:
        def encode(self, record, processing_class):
            prompt, completion = [1], [2]
            yield {"input_ids": prompt, "labels": completion, "revision": "custom"}

    chats = chat_dataset_from_records(
        [
            {
                "messages": [
                    {"role": "user", "content": "input"},
                    {"role": "assistant", "content": "output"},
                ],
                "trace_id": "trace",
                "reward": 0.75,
            }
        ]
    )
    result = datum_dataset_from_chat_dataset(chats, Processor(), CustomAdapter())
    assert json.loads(result[0]["datum_json"]) == {
        "input_ids": [1],
        "labels": [2],
        "revision": "custom",
    }
    assert result[0]["trace_id"] == "trace" and result[0]["reward"] == 0.75
