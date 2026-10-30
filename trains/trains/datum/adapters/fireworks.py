from collections.abc import Iterator, Mapping
from typing import Any

from fireworks.training import sdk

from trains.datum.adapters.capabilities import require_capability
from trains.datum.adapters.rendering import native_datum, render_datums
from trains.formats.tokenization import encoded_datum


class FireworksDatumAdapter:
    """Build native fireworks datums from records or exact native renderer output."""

    provider = "fireworks"

    def from_tokens(self, prompt: list[int], completion: list[int]) -> sdk.Datum:
        """Build a native datum with next-token targets and assistant-only weights."""
        value = encoded_datum(prompt, completion)
        return sdk.Datum(
            model_input=sdk.ModelInput(
                chunks=[sdk.EncodedTextChunk(tokens=prompt + completion[:-1])]
            ),
            loss_fn_inputs={
                name: sdk.TensorData(**tensor) for name, tensor in value["loss_fn_inputs"].items()
            },
        )

    def encode(self, record: Mapping[str, Any], processing_class: Any) -> Iterator[sdk.Datum]:
        """Yield native datums rendered with the supplied tokenizer or processor."""
        for value in render_datums(record, processing_class):
            for chunk in value["model_input"]["chunks"]:
                if chunk["type"] != "encoded_text":
                    require_capability(
                        self.provider, "audio" if chunk["type"] == "dmel" else "image"
                    )
            yield native_datum(value, sdk)
