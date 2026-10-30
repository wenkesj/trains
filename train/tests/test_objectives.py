import pytest

from train.models import Batch, Datum, ModelInput, Tensor, TextChunk
from train.objectives import validate_batch


@pytest.mark.parametrize("shape", [None, [3], [3, 1], [3, 2]])
def test_training_requires_one_target_per_position(shape):
    count = 6 if shape == [3, 2] else 3
    batch = Batch(
        data=[
            Datum(
                model_input=ModelInput(chunks=[TextChunk(tokens=[1, 4, 5])]),
                loss_fn_inputs={
                    "target_tokens": Tensor(data=[6] * count, dtype="int64", shape=shape),
                    "weights": Tensor(data=[1.0] * count, dtype="float32", shape=shape),
                },
            )
        ]
    )
    if shape is not None and len(shape) > 1:
        with pytest.raises(ValueError, match="one-dimensional"):
            validate_batch(batch, vocab_size=16, context=32)
    else:
        validate_batch(batch, vocab_size=16, context=32)
