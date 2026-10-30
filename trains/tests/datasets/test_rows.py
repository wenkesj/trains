import pytest

from trains.datasets import chat_dataset_from_records
from trains.datum.adapters.tinker import TinkerDatumAdapter


def test_invalid_tensors_and_rewards_fail_closed():
    with pytest.raises(ValueError, match="Reward"):
        chat_dataset_from_records(
            [
                {
                    "messages": [
                        {"role": "user", "content": "go"},
                        {"role": "assistant", "content": "north"},
                    ],
                    "reward": float("nan"),
                }
            ]
        )
    with pytest.raises(ValueError, match="nonempty"):
        TinkerDatumAdapter().from_tokens([], [1])
