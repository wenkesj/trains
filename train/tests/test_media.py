import base64
import io

import pytest
import tinker
import torch
from baseten import loops
from fireworks.training import sdk
from PIL import Image
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

from train.backend import LocalTrainer
from train.models import AdamParams, Batch, ModelSpec, Tensor
from train.runtime import Runtime


@pytest.fixture
def vision_case(tmp_path, monkeypatch):
    config = Qwen2VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "embed_dim": 16,
            "hidden_size": 16,
            "num_heads": 2,
            "mlp_ratio": 2,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 1,
        },
        image_token_id=30,
        video_token_id=29,
        vision_start_token_id=28,
        vision_end_token_id=27,
    )
    model_path = tmp_path / "vision"
    Qwen2VLForConditionalGeneration(config).save_pretrained(model_path)

    class ImageProcessor:
        merge_size = 1

        def __call__(self, images, **kwargs):
            return {"pixel_values": torch.ones(4, 12), "image_grid_thw": torch.tensor([[1, 2, 2]])}

    monkeypatch.setattr(
        "train.media.AutoImageProcessor.from_pretrained",
        lambda *a, **k: ImageProcessor(),
    )
    image = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(image, format="PNG")
    batch = Batch.model_validate(
        {
            "data": [
                {
                    "model_input": {
                        "chunks": [
                            {"tokens": [1, 28]},
                            {
                                "type": "image",
                                "data": base64.b64encode(image.getvalue()).decode(),
                                "format": "png",
                                "expected_tokens": 4,
                            },
                            {"tokens": [27, 2]},
                        ]
                    },
                    "loss_fn_inputs": {
                        "target_tokens": {
                            "data": [28, 0, 0, 0, 0, 27, 2, 3],
                            "dtype": "int64",
                            "shape": [8],
                        },
                        "weights": {
                            "data": [0, 0, 0, 0, 0, 0, 0, 1],
                            "dtype": "float32",
                            "shape": [8],
                        },
                    },
                }
            ]
        }
    )
    spec = ModelSpec(base_model="vision", base_path=model_path, rank=2, train_unembed=False)
    return spec, batch


def test_real_tiny_vision_model_trains_and_restores(vision_case, tmp_path):
    spec, batch = vision_case
    trainer = LocalTrainer(spec, runtime=Runtime(kernels="reference", mlp_chunk_size=0))
    result = trainer.forward(batch, True)
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert trainer.optim_step(AdamParams())["metrics"]["step"] == 1
    trainer.save(tmp_path / "checkpoint", sampler=False)
    restored = LocalTrainer(spec, runtime=Runtime(kernels="reference", mlp_chunk_size=0))
    restored.load(tmp_path / "checkpoint", optimizer=True)
    assert restored.step == 1
    batch.data[0].model_input.chunks[1].expected_tokens = 5
    with pytest.raises(ValueError):
        restored.forward(batch, False)


@pytest.mark.parametrize("provider", ["tinker", "spindle", "fireworks", "baseten"])
def test_native_image_training_contract(vision_case, training_server, model_path, provider):
    spec, batch = vision_case
    model = Qwen2VLForConditionalGeneration.from_pretrained(spec.base_path, local_files_only=True)
    model.save_pretrained(model_path)
    module = {"tinker": tinker, "spindle": tinker, "fireworks": sdk, "baseten": loops}[provider]
    types = tinker.types if module is tinker else module
    service = module.ServiceClient(base_url=training_server, api_key="tml-local")
    training = service.create_lora_training_client(base_model="tiny", rank=2)
    value = batch.data[0]
    native = types.Datum(
        model_input=types.ModelInput.model_validate(value.model_input.model_dump()),
        loss_fn_inputs={
            key: types.TensorData(**tensor.model_dump())
            for key, tensor in value.loss_fn_inputs.items()
        },
    )
    result = training.forward_backward([native], "cross_entropy").result(timeout=20)
    assert result.loss_fn_outputs[0]["logprobs"].shape == [8]
    assert training.optim_step(types.AdamParams()).result(timeout=20).metrics["step"] == 1
    if module is tinker:
        service.close("success").result()
    elif module is loops:
        training.close()
    else:
        service.close()


@pytest.mark.parametrize("loss_fn", ["cross_entropy", "ppo", "importance_sampling"])
def test_selected_vision_targets_preserve_full_forward_loss_and_gradients(vision_case, loss_fn):
    spec, batch = vision_case
    batch.loss_fn = loss_fn
    if loss_fn != "cross_entropy":
        weights = batch.data[0].loss_fn_inputs["weights"].data
        batch.data[0].loss_fn_inputs.update(
            logprobs=Tensor(dtype="float32", data=[-3.0] * len(weights)),
            advantages=Tensor(dtype="float32", data=[0.5 * weight for weight in weights]),
        )
    full = LocalTrainer(spec, runtime=Runtime(kernels="reference", mlp_chunk_size=0))
    selected = LocalTrainer(spec, runtime=Runtime(kernels="reference", mlp_chunk_size=0))
    reference = full.forward(batch, True)
    optimized = selected.forward(batch, True, return_outputs=False)
    assert optimized["loss"] == pytest.approx(reference["loss"], abs=1e-5)
    for (name, left), (other, right) in zip(
        full.model.named_parameters(), selected.model.named_parameters(), strict=True
    ):
        assert name == other
        if left.grad is not None:
            assert right.grad is not None, name
            torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=1e-4)
