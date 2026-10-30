import json

import pandas as pd
from tinker import types

from trains.datum.adapters.tinker import TinkerDatumAdapter


def raw_datum():
    return types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"target_tokens": types.TensorData(data=[2, 3], dtype="int64", shape=[2])},
    )


def read_datum(value):
    value = json.loads(value)
    return types.Datum(
        model_input=types.ModelInput.model_validate(value["model_input"]),
        loss_fn_inputs={
            name: types.TensorData(**tensor) for name, tensor in value["loss_fn_inputs"].items()
        },
    )


def span_frame():
    return pd.DataFrame(
        [
            {
                "context.trace_id": "trace-native",
                "context.span_id": "root",
                "attributes.openinference.span.kind": "AGENT",
            },
            {
                "context.trace_id": "trace-native",
                "context.span_id": "llm",
                "parent_id": "root",
                "attributes.openinference.span.kind": "LLM",
                "attributes.input.value": json.dumps(
                    {"messages": [{"role": "user", "content": "go"}]}
                ),
                "attributes.output.value": json.dumps({"role": "assistant", "content": "north"}),
            },
        ]
    ).fillna("")


def datum(index=0):
    return TinkerDatumAdapter().from_tokens([index], [index + 1])


def shards_factory(shards):
    for shard in shards:
        yield datum(shard)


class Processor:
    def __init__(self, offset=0):
        self.offset = offset

    def encode(self, text, **kwargs):
        return [self.offset + ord(char) for char in text]

    def apply_chat_template(self, messages, *, tokenize=True, **kwargs):
        text = "".join(message.get("content") or "" for message in messages)
        return self.encode(text) if tokenize else text
