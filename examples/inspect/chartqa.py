import base64
import hashlib
import io
import math

from datasets import Dataset, load_dataset
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState

CHARTQA_DATASET = "HuggingFaceM4/ChartQA"
CHARTQA_REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"


def load_charts(split: str, *, offset: int = 0, limit: int = 4) -> MemoryDataset:
    """Load original ChartQA images and labels through the native HF dataset cache.

    Args:
        split: Native dataset split; train and test remain separate.
        offset: First row in the requested split.
        limit: Number of original chart questions to materialize.

    Returns:
        Native Inspect samples with portable images and pinned dataset provenance.
    """
    if offset < 0 or limit < 1:
        raise ValueError("Use a nonnegative offset and a positive chart limit")
    rows = load_dataset(CHARTQA_DATASET, revision=CHARTQA_REVISION, split=split)
    if not isinstance(rows, Dataset):
        raise TypeError("Select one ChartQA split")
    samples = []
    for index, row in enumerate(
        rows.select(range(offset, min(offset + limit, len(rows)))), start=offset
    ):
        buffer = io.BytesIO()
        row["image"].convert("RGB").save(buffer, format="PNG")
        raw = buffer.getvalue()
        samples.append(
            Sample(
                id=f"{split}-{index}",
                input=[
                    ChatMessageUser(
                        content=[
                            ContentImage(
                                image="data:image/png;base64," + base64.b64encode(raw).decode()
                            ),
                            ContentText(
                                text=row["query"]
                                + "\nAnswer with only the answer, without explanation."
                            ),
                        ]
                    )
                ],
                target=row["label"],
                metadata={
                    "dataset": CHARTQA_DATASET,
                    "revision": CHARTQA_REVISION,
                    "split": split,
                    "row_index": index,
                    "image_sha256": hashlib.sha256(raw).hexdigest(),
                },
            )
        )
    if not samples:
        raise ValueError("The selected chart slice is empty")
    return MemoryDataset(samples=samples, name="chartqa-" + split)


@scorer(metrics=[accuracy()])
def relaxed_accuracy() -> Scorer:
    """Score ChartQA answers with case-insensitive matching and 5% numeric tolerance."""

    async def score(state: TaskState, target: Target) -> Score:
        prediction = state.output.completion.strip()
        return Score(
            value=int(any(_matches(value, prediction) for value in target)), answer=prediction
        )

    return score


def _matches(target: str, prediction: str) -> bool:
    """Compare final answers without accepting unrelated numbers in explanations.

    Examples:
        >>> _matches("100", "104")
        True
        >>> _matches("100", "106")
        False
        >>> _matches("0", "0.0")
        True
        >>> _matches("0.39", "39%")
        True
        >>> _matches("Yes", "yes")
        True
        >>> _matches("6", "There are 6 values")
        False
    """
    left, right = _number(target), _number(prediction)
    if left is not None and right is not None:
        return abs(right - left) <= abs(left) * 0.05
    return target.strip().casefold() == prediction.strip().casefold()


def _number(value: str) -> float | None:
    """Read finite numeric answers, interpreting percent notation consistently."""
    try:
        number = float(value.removesuffix("%")) / (100 if value.endswith("%") else 1)
        return number if math.isfinite(number) else None
    except ValueError:
        return None
