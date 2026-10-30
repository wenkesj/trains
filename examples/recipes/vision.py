import copy
import io
from typing import Any

import torch
from PIL import Image
from tinker import types
from transformers import AutoImageProcessor

from recipes.models import DEFAULT_MODEL, LFM3_REVISION, load_tokenizer
from trains.datasets.assets import read_asset
from trains.formats.records import validate_record
from trains.formats.tokenization import encode


class LiquidVisionRenderer:
    """Preserve the native Liquid template, image grid, and assistant loss mask."""

    supports_message_weights = True
    supports_rendering_options = True

    def __init__(self, model: str = DEFAULT_MODEL, *, trust_remote_code: bool = False) -> None:
        """Load the served tokenizer and image processor from the same revision.

        Args:
            model: Liquid Omni model identity or a local snapshot directory.
            trust_remote_code: Allow custom code from an explicitly selected snapshot.
        """
        self.tokenizer = load_tokenizer(model, trust_remote_code=trust_remote_code)
        self.image_processor = AutoImageProcessor.from_pretrained(
            model,
            trust_remote_code=trust_remote_code or model == DEFAULT_MODEL,
            **({"revision": LFM3_REVISION} if model == DEFAULT_MODEL else {}),
        )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image|>")

    def build_supervised_examples(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> list[tuple[types.ModelInput, torch.Tensor]]:
        """Expand image placeholders while retaining exact template supervision.

        Args:
            messages: Complete conversation with resolved image assets.
            tools: Original function schemas supplied during collection.
            chat_template_kwargs: Original inference template settings.

        Returns:
            One native model input and its unshifted assistant supervision mask.
        """
        native, images = _image_messages(messages)
        tokens, weights = encode(
            validate_record(
                {"messages": native, "tools": tools, "chat_template_kwargs": chat_template_kwargs}
            ),
            self.tokenizer,
        )
        grids = (
            self.image_processor(images=images, return_tensors="pt")["image_grid_thw"]
            if images
            else []
        )
        counts = [int(grid.prod()) // self.image_processor.merge_size**2 for grid in grids]
        chunks: list[types.EncodedTextChunk | types.ImageChunk] = []
        expanded, start, cursor = [], 0, 0
        for index, token in enumerate(tokens):
            if token != self.image_token_id:
                expanded.append(weights[index])
                continue
            if cursor >= len(images) or weights[index]:
                raise ValueError("Image placeholders must align with unsupervised input images")
            if index > start:
                chunks.append(types.EncodedTextChunk(tokens=tokens[start:index]))
            buffer = io.BytesIO()
            images[cursor].save(buffer, format="PNG")
            chunks.append(
                types.ImageChunk(
                    data=buffer.getvalue(), format="png", expected_tokens=counts[cursor]
                )
            )
            expanded.extend([0.0] * counts[cursor])
            cursor += 1
            start = index + 1
        if cursor != len(images):
            raise ValueError("Chat template omitted an input image")
        if start < len(tokens):
            chunks.append(types.EncodedTextChunk(tokens=tokens[start:]))
        return [(types.ModelInput(chunks=chunks), torch.tensor(expanded, dtype=torch.float32))]


def _image_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Normalize OpenAI image parts without changing their order or text content."""
    result, images = copy.deepcopy(messages), []
    for message in result:
        if not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            kind = part["type"]
            if kind in {"text", "input_text", "output_text"}:
                part["type"] = "text"
            elif kind in {"image", "image_url", "input_image"}:
                uri = part.get("image_url") or part.get("image") or part.get("url")
                if isinstance(uri, dict):
                    uri = uri["url"]
                if not isinstance(uri, str):
                    raise ValueError("Image content requires a resolved local or data URI")
                data, _ = read_asset(uri)
                with Image.open(io.BytesIO(data)) as image:
                    images.append(image.convert("RGB"))
                part.clear()
                part.update(type="image")
            else:
                raise ValueError(f"Liquid vision rendering does not support {kind!r} content")
    return result, images
