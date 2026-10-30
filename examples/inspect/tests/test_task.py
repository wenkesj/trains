from inspect_ai import Task
from inspect_ai.dataset import Sample

import task as example_task


def test_train_and_heldout_slices_keep_native_ids_and_fixed_judge(monkeypatch):
    calls = []

    def native_task(**kwargs):
        calls.append(kwargs)
        return Task(
            dataset=[
                Sample(id=str(i), input="Chart", target="Value") for i in range(kwargs["limit"])
            ]
        )

    monkeypatch.setattr(example_task, "_chartography", native_task)
    training = example_task.chartography(offset=0, limit=2)
    heldout = example_task.chartography(offset=2, limit=2)
    assert [sample.id for sample in training.dataset] == ["0", "1"]
    assert [sample.id for sample in heldout.dataset] == ["2", "3"]
    assert all(call["judge_model"] == "openai-api/judge/LiquidAI/LFM3-Omni-2B" for call in calls)
