import base64
import copy
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from trains.datasets.assets import resolve_assets
from trains.datasets.serialization import datum_mapping
from trains.formats.records import validate_record
from trains.formats.tokenization import conversation_datum, encode


def render_datums(record: Mapping[str, Any], processor: Any) -> Iterator[Mapping[str, Any]]:
    """Delegate native multimodal rendering; text retains HF prefix validation."""
    record = resolve_assets(record)
    record = {key: value for key, value in record.items() if value is not None}
    record = validate_record(record)
    if hasattr(processor, "build_supervised_examples"):
        if any(message.get("weight", 1) != 1 for message in record["messages"]) and not getattr(
            processor, "supports_message_weights", False
        ):
            raise ValueError(
                "Per-message supervision weights require an HF tokenizer or a custom native adapter"
            )
        yield from _renderer_datums(record, processor)
        return
    if any(isinstance(message.get("content"), list) for message in record["messages"]):
        raise ValueError(
            "Multimodal data requires a native renderer with build_supervised_examples"
        )
    yield conversation_datum(*encode(record, processor))


def native_datum(value: Any, sdk: Any) -> Any:
    """Reconstruct a provider SDK datum from a serialized native mapping."""
    value = copy.deepcopy(value)
    for chunk in value["model_input"]["chunks"]:
        for key in ["data", "dmel"]:
            if key in chunk and isinstance(chunk[key], str):
                chunk[key] = base64.b64decode(chunk[key], validate=True)
    return sdk.Datum(
        model_input=sdk.ModelInput.model_validate(value["model_input"]),
        loss_fn_inputs={
            key: sdk.TensorData(**tensor) for key, tensor in value["loss_fn_inputs"].items()
        },
    )


def _chunk_targets(
    chunks: Sequence[Mapping[str, Any]], lengths: Sequence[int] | None = None
) -> list[int]:
    """Extract the target token sequence from rendered model-input chunks."""
    result = []
    for index, chunk in enumerate(chunks):
        if chunk["type"] == "encoded_text":
            result.extend(chunk["tokens"])
        else:
            count = lengths[index] if lengths else chunk.get("expected_tokens")
            if not isinstance(count, int) or count <= 0:
                raise ValueError("Media rendering requires exact expected_tokens")
            result.extend([0] * count)
    return result


def _renderer_datums(record: Mapping[str, Any], processor: Any) -> Iterator[Mapping[str, Any]]:
    """Expand renderer output into native datums with aligned assistant weights."""
    messages = record["messages"]
    options = {key: record[key] for key in ("tools", "chat_template_kwargs") if record.get(key)}
    if options and not getattr(processor, "supports_rendering_options", False):
        raise ValueError("Native renderer must explicitly support tools and chat_template_kwargs")
    for model_input, weights in processor.build_supervised_examples(messages, **options):
        chunks = datum_mapping({"chunks": model_input.chunks})["chunks"]
        targets = _chunk_targets(chunks, [chunk.length for chunk in model_input.chunks])
        values = weights.tolist() if hasattr(weights, "tolist") else list(weights)
        cursor = 0
        for chunk, native_chunk in zip(chunks, model_input.chunks, strict=True):
            if chunk["type"] != "encoded_text" and any(
                values[cursor : cursor + native_chunk.length]
            ):
                raise ValueError("Media positions must not carry text loss")
            cursor += native_chunk.length
        if len(values) != len(targets):
            raise ValueError("Renderer weights must align with native model positions")
        if chunks[-1]["type"] != "encoded_text" or not chunks[-1]["tokens"]:
            raise ValueError("Supervised output must end in a text token")
        chunks[-1]["tokens"] = chunks[-1]["tokens"][:-1]
        yield {
            "model_input": {"chunks": [c for c in chunks if c.get("tokens") != []]},
            "loss_fn_inputs": {
                "target_tokens": {
                    "data": targets[1:],
                    "dtype": "int64",
                    "shape": [len(targets) - 1],
                },
                "weights": {
                    "data": values[1:],
                    "dtype": "float32",
                    "shape": [len(targets) - 1],
                },
            },
        }
