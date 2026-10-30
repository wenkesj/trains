import base64
import io
import os

import pytest
from PIL import Image
from recipes.vision import LiquidVisionRenderer

from trains.datum.adapters.rendering import render_datums


@pytest.fixture
def renderer():
    snapshot = os.environ.get("TRAIN_TEST_LFM3_OMNI_SNAPSHOT")
    if not snapshot:
        pytest.skip("requires the pinned Liquid Omni tokenizer and image processor")
    return LiquidVisionRenderer(snapshot, trust_remote_code=True)


def test_native_omni_images_keep_context_masked_and_supervise_only_the_final_action(renderer):
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128), "red").save(buffer, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}]},
        {"role": "assistant", "content": "Earlier answer", "weight": 0},
        {"role": "user", "content": "What color?"},
        {"role": "assistant", "content": "Red"},
    ]
    datum = next(
        render_datums(
            {
                "messages": messages,
                "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
            },
            renderer,
        )
    )
    chunks = datum["model_input"]["chunks"]
    targets = datum["loss_fn_inputs"]["target_tokens"]["data"]
    weights = datum["loss_fn_inputs"]["weights"]["data"]
    image = next(chunk for chunk in chunks if chunk["type"] == "image")
    assert image["expected_tokens"] == 64
    assert base64.b64decode(image["data"]) == buffer.getvalue()
    assert (
        len(targets)
        == len(weights)
        == sum(
            len(chunk["tokens"]) if chunk["type"] == "encoded_text" else chunk["expected_tokens"]
            for chunk in chunks
        )
    )
    assert (
        renderer.tokenizer.decode(
            [token for token, weight in zip(targets, weights, strict=True) if weight]
        )
        == "Red<|im_end|>"
    )


def test_omni_rejects_unsupported_media_instead_of_training_on_dropped_content(renderer):
    with pytest.raises(ValueError, match="does not support 'audio'"):
        renderer.build_supervised_examples(
            [
                {"role": "user", "content": [{"type": "audio", "audio": "unused"}]},
                {"role": "assistant", "content": "Transcript"},
            ]
        )
