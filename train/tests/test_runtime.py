import os
from types import SimpleNamespace

import pytest
import torch

from train.backend import LocalTrainer
from train.models import AdamParams, Batch, ModelSpec
from train.runtime import Runtime


def test_runtime_does_not_silently_fall_back_to_cpu(model_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA requested"):
        Runtime(device="cuda").load(ModelSpec(base_model="tiny", base_path=model_path, rank=2))


@pytest.mark.skipif(not os.environ.get("TRAIN_TEST_CUDA"), reason="explicit GPU smoke test")
def test_cuda_bfloat16_checkpointed_training(model_path, tmp_path):
    spec = ModelSpec(base_model="tiny", base_path=model_path, rank=2, train_unembed=False)
    runtime = Runtime(device="cuda:0", dtype="bfloat16", gradient_checkpointing=True)
    batch = Batch.model_validate(
        {
            "data": [
                {
                    "model_input": {"chunks": [{"tokens": [1, 4, 5]}]},
                    "loss_fn_inputs": {"target_tokens": {"data": [4, 5, 6], "dtype": "int64"}},
                }
            ]
        }
    )
    trainer = LocalTrainer(spec, runtime=runtime)
    trainer.forward(batch, backward=True)
    assert trainer.optim_step(AdamParams())["metrics"]["step"] == 1
    trainer.save(tmp_path / "checkpoint", sampler=False)
    restored = LocalTrainer(spec, runtime=runtime)
    restored.load(tmp_path / "checkpoint", optimizer=True)
    restored.forward(batch, backward=True)
    assert restored.optim_step(AdamParams())["metrics"]["step"] == 2


@pytest.mark.parametrize(
    "device,dtype,available,expected",
    [
        ("cuda", "bfloat16", True, "flash_attention_4"),
        ("cuda", "bfloat16", False, "sdpa"),
        ("cuda", "float32", True, "sdpa"),
        ("cpu", "bfloat16", True, "sdpa"),
    ],
)
def test_auto_attention_selection_uses_device_precision_and_installed_backend(
    monkeypatch, device, dtype, available, expected
):
    monkeypatch.setattr("train.runtime.is_flash_attn_4_available", lambda: available)
    assert Runtime(dtype=dtype)._attention_implementation(torch.device(device)) == expected


def test_explicit_flash_attention_requires_supported_runtime(monkeypatch):
    monkeypatch.setattr("train.runtime.is_flash_attn_4_available", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA bfloat16"):
        Runtime(dtype="bfloat16", attention_backend="flash_attention_4")._attention_implementation(
            torch.device("cuda")
        )


def test_attention_follows_cuda_cpu_offload_roundtrip(monkeypatch):
    class Model:
        _supports_flash_attn = True

        def __init__(self):
            self.config = SimpleNamespace(_attn_implementation="sdpa")
            self.backends = []

        def to(self, device):
            return self

        def modules(self):
            return []

        def set_attn_implementation(self, value):
            self.config._attn_implementation = value
            self.backends.append(value)

    monkeypatch.setattr("train.runtime.is_flash_attn_4_available", lambda: True)
    model = Model()
    runtime = Runtime(dtype="bfloat16")
    for device in ("cuda", "cpu", "cuda"):
        assert runtime.place(model, device) is model
    assert model.backends == ["flash_attention_4", "sdpa", "flash_attention_4"]
