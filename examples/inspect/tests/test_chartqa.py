import base64
import hashlib
import io

import pytest
from datasets import Dataset, Features, Image, List, Value
from inspect_ai.model import ChatMessageUser, ContentImage
from PIL import Image as PILImage

import chartqa


@pytest.mark.parametrize(
    ("target", "prediction", "correct"),
    [
        ("100", "105", True),
        ("100", "106", False),
        ("0", "0.0", True),
        ("0", "0.001", False),
        ("0.39", "39%", True),
        ("Yes", "yes", True),
        ("6", "There are 6 values", False),
        ("1", "inf", False),
    ],
)
def test_chartqa_scores_final_answers_with_numeric_tolerance(target, prediction, correct):
    assert chartqa._matches(target, prediction) is correct


def test_chartqa_keeps_original_image_labels_slice_and_provenance(monkeypatch):
    rows = Dataset.from_dict(
        {
            "image": [PILImage.new("RGB", (8, 8), color) for color in ["red", "blue"]],
            "query": ["First question?", "Second question?"],
            "label": [["Yes"], ["No"]],
        },
        features=Features(
            {"image": Image(), "query": Value("string"), "label": List(Value("string"))}
        ),
    )
    monkeypatch.setattr(chartqa, "load_dataset", lambda *args, **kwargs: rows)
    (sample,) = chartqa.load_charts("train", offset=1, limit=1)
    assert sample.id == "train-1"
    assert sample.target == ["No"]
    assert sample.metadata is not None
    assert sample.metadata["revision"] == chartqa.CHARTQA_REVISION
    assert sample.metadata["row_index"] == 1
    assert isinstance(sample.input, list)
    message = sample.input[0]
    assert isinstance(message, ChatMessageUser) and isinstance(message.content, list)
    content = message.content[0]
    assert isinstance(content, ContentImage)
    raw = base64.b64decode(content.image.split(",", 1)[1])
    assert hashlib.sha256(raw).hexdigest() == sample.metadata["image_sha256"]
    with PILImage.open(io.BytesIO(raw)) as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)
