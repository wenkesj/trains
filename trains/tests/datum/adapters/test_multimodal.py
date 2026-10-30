import pytest
from tinker import types

from trains.datasets import chat_dataset_from_records, datum_dataset_from_chat_dataset
from trains.datasets.serialization import datum_mapping
from trains.datum.adapters.baseten import BasetenDatumAdapter
from trains.datum.adapters.fireworks import FireworksDatumAdapter
from trains.datum.adapters.spindle import SpindleDatumAdapter
from trains.datum.adapters.tinker import TinkerDatumAdapter


class Renderer:
    def build_supervised_examples(self, messages):
        yield (
            types.ModelInput(
                chunks=[
                    types.EncodedTextChunk(tokens=[1]),
                    types.ImageChunk(data=b"png", format="png", expected_tokens=2),
                    types.EncodedTextChunk(tokens=[2, 3]),
                ]
            ),
            [0, 0, 0, 0, 1],
        )


@pytest.mark.parametrize(
    "adapter",
    [TinkerDatumAdapter(), SpindleDatumAdapter(), BasetenDatumAdapter(), FireworksDatumAdapter()],
)
def test_native_image_chunks_survive_all_provider_datums(adapter):
    source = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}}
                ],
            },
            {"role": "assistant", "content": "answer"},
        ]
    }
    datum = next(iter(adapter.encode(source, Renderer())))
    value = datum_mapping(datum)
    assert value["model_input"]["chunks"][1]["type"] == "image"
    assert value["loss_fn_inputs"]["weights"]["data"] == [0, 0, 0, 1]
    assert (
        len(
            datum_dataset_from_chat_dataset(
                chat_dataset_from_records([source]), Renderer(), adapter
            )
        )
        == 1
    )
