from datasets import load_from_disk
from trains.datasets import chat_dataset_from_records


def test_chat_json_features_preserve_heterogeneous_tools_and_arguments(tmp_path):
    records = [
        {
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup",
                                "arguments": {"city": "Paris", "years": [2024, 2025]},
                            }
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "years": {"type": "array", "items": {"type": "integer"}},
                            },
                        },
                    },
                }
            ],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        {
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "north"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "north"},
            ]
        },
    ]
    records[0]["messages"][-1]["reasoning_content"] = "Choose the lookup tool."
    dataset = chat_dataset_from_records(records)
    dataset.save_to_disk(tmp_path / "chat")
    restored = load_from_disk(tmp_path / "chat")
    assert restored[:] == dataset[:]
    assert restored[0]["tools"] == records[0]["tools"]
    assert restored[0]["messages"] == records[0]["messages"]
    assert restored[1]["messages"] == records[1]["messages"]
    assert restored[2]["messages"][0]["content"] == "go"
