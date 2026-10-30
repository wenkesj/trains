import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

from train.attention import _head_chunks


def test_gdn_head_chunks_preserve_values_and_gradients():
    inputs = [torch.randn(1, 8, 4, 8, requires_grad=True) for _ in range(3)]
    inputs += [(-torch.rand(1, 8, 4)).requires_grad_(), torch.rand(1, 8, 4, requires_grad=True)]
    expected, _ = torch_chunk_gated_delta_rule(*inputs, use_qk_l2norm_in_kernel=True)
    expected.square().sum().backward()
    gradients = [x.grad.clone() for x in inputs]
    for x in inputs:
        x.grad = None
    actual, state = _head_chunks(
        torch_chunk_gated_delta_rule, 2, *inputs, use_qk_l2norm_in_kernel=True
    )
    actual.square().sum().backward()
    assert state is None
    assert torch.allclose(actual, expected, atol=1e-05, rtol=0.0001)
    for x, gradient in zip(inputs, gradients, strict=True):
        assert torch.allclose(x.grad, gradient, atol=1e-05, rtol=0.0001)
