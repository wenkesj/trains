import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated

from train.backend import LocalTrainer
from train.mlp import _chunked_gated_norm, chunk_mlps
from train.models import Batch, ModelSpec
from train.runtime import Runtime


def test_chunked_gated_norm_preserves_values_and_gradients():
    norm = Qwen3_5RMSNormGated(8)
    values = torch.randn(23, 8, requires_grad=True)
    gate = torch.randn(23, 8, requires_grad=True)
    expected = norm(values, gate)
    expected.square().sum().backward()
    gradients = [x.grad.clone() for x in (values, gate, norm.weight)]
    for x in (values, gate, norm.weight):
        x.grad = None
    actual = _chunked_gated_norm(norm.forward, 4, values, gate)
    actual.square().sum().backward()
    assert torch.allclose(actual, expected)
    for x, gradient in zip((values, gate, norm.weight), gradients, strict=True):
        assert torch.allclose(x.grad, gradient, atol=1e-05, rtol=0.0001)


def test_chunk_wrappers_restore_after_kernel_forward_replacement():
    norm = Qwen3_5RMSNormGated(8)
    original = norm.forward
    chunk_mlps(norm, 1)
    wrapped = norm.forward
    chunk_mlps(norm, 1)
    assert norm.forward is wrapped
    norm.forward = original
    chunk_mlps(norm, 1)
    assert norm.forward.func is _chunked_gated_norm


def test_chunked_offloaded_mlp_preserves_gradients_and_checkpoint_keys(model_path):
    spec = ModelSpec(base_model="tiny", base_path=model_path, rank=2, train_unembed=False)
    plain = LocalTrainer(spec, runtime=Runtime(mlp_chunk_size=0))
    chunked = LocalTrainer(
        spec,
        runtime=Runtime(gradient_checkpointing=True, checkpoint_offload=True, mlp_chunk_size=2),
    )
    batch = Batch.model_validate(
        {
            "loss_fn": "cross_entropy",
            "data": [
                {
                    "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 4, 5]}]},
                    "loss_fn_inputs": {
                        "target_tokens": {"data": [4, 5, 6], "dtype": "int64"},
                        "weights": {"data": [0, 1, 1], "dtype": "float32"},
                    },
                }
            ],
        }
    )
    expected = plain.forward(batch, backward=True, return_outputs=False)
    actual = chunked.forward(batch, backward=True, return_outputs=False)
    assert actual["loss"] == pytest.approx(expected["loss"], abs=1e-05)
    assert plain.model.state_dict().keys() == chunked.model.state_dict().keys()
    for (name, left), (_, right) in zip(
        plain.model.named_parameters(), chunked.model.named_parameters(), strict=True
    ):
        if left.grad is not None:
            assert right.grad is not None, name
            assert torch.allclose(left.grad, right.grad, atol=1e-05, rtol=0.0001), name
