import pytest
import torch

from train.backend import LocalTrainer, _sampled_logprobs
from train.models import AdamParams, Batch, ModelSpec, SamplingParams
from train.runtime import Runtime
from train.sampling import sample


def test_lora_gradient_reduces_teacher_forced_loss(model_path):
    trainer = LocalTrainer(
        ModelSpec(base_model="tiny", base_path=model_path, rank=2, train_unembed=False),
        runtime=Runtime(),
    )
    batch = Batch.model_validate(
        {
            "data": [
                {
                    "model_input": {"chunks": [{"tokens": [1, 4, 5]}]},
                    "loss_fn_inputs": {"target_tokens": {"dtype": "int64", "data": [4, 5, 6]}},
                }
            ]
        }
    )
    before = trainer.forward(batch, backward=False)["loss"]
    for _ in range(3):
        trainer.forward(batch, backward=True)
        trainer.optim_step(AdamParams(learning_rate=0.01), normalization="num_loss_tokens")
    assert trainer.forward(batch, backward=False)["loss"] < before


def test_failed_backward_preserves_existing_accumulated_gradients(model_path):
    trainer = LocalTrainer(
        ModelSpec(base_model="tiny", base_path=model_path, rank=2, train_unembed=False),
        runtime=Runtime(),
    )
    datum = {
        "model_input": {"chunks": [{"tokens": [1, 4, 5]}]},
        "loss_fn_inputs": {"target_tokens": {"dtype": "int64", "data": [4, 5, 6]}},
    }
    trainer.forward(Batch.model_validate({"data": [datum]}), backward=True)
    before = {
        name: parameter.grad.clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.grad is not None
    }
    first = {
        **datum,
        "loss_fn_inputs": {
            **datum["loss_fn_inputs"],
            "logprobs": {"dtype": "float32", "data": [-3.0, -3.0, -3.0]},
            "advantages": {"dtype": "float32", "data": [1.0, 1.0, 1.0]},
        },
    }
    second = {
        **first,
        "loss_fn_inputs": {
            **first["loss_fn_inputs"],
            "logprobs": {"dtype": "float32", "data": [-1e30] * 3},
        },
    }
    with pytest.raises(ValueError, match="finite"):
        trainer.forward(
            Batch.model_validate({"data": [first, second], "loss_fn": "importance_sampling"}),
            backward=True,
        )
    assert trainer.sequences == 1
    for name, parameter in trainer.model.named_parameters():
        if name in before:
            assert torch.equal(parameter.grad, before[name])


def test_chunked_scoring_matches_full_softmax_and_gradients():
    torch.manual_seed(4)
    logits = torch.randn(321, 67, requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    targets = torch.randint(67, (321,))
    weights = torch.randn(321)
    actual = _sampled_logprobs(logits, targets)
    expected = reference.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, reference.grad)


@pytest.mark.parametrize("objective", ["cross_entropy", "ppo", "importance_sampling"])
def test_selected_projection_matches_full_loss_gradients_and_normalization(model_path, objective):
    spec = ModelSpec(base_model="tiny", base_path=model_path, rank=2, train_unembed=False)
    full = LocalTrainer(spec, runtime=Runtime())
    selected = LocalTrainer(spec, runtime=Runtime())
    datum = {
        "model_input": {"chunks": [{"tokens": [1, 4, 5, 6]}]},
        "loss_fn_inputs": {
            "target_tokens": {"dtype": "int64", "data": [4, 5, 6, 7]},
            "weights": {"dtype": "float32", "data": [0.0, 0.0, 1.0, 1.0]},
        },
    }
    if objective != "cross_entropy":
        datum["loss_fn_inputs"].update(
            logprobs={"dtype": "float32", "data": [-3.5] * 4},
            advantages={"dtype": "float32", "data": [0.0, 0.0, 0.5, -0.5]},
        )
    batch = Batch.model_validate({"data": [datum], "loss_fn": objective})
    reference = full.forward(batch, True)
    result = selected.forward(batch, True, return_outputs=False)
    assert result["loss"] == pytest.approx(reference["loss"], abs=1e-6)
    assert selected.loss_tokens == full.loss_tokens == 2
    for (name, a), (other, b) in zip(
        full.model.named_parameters(), selected.model.named_parameters(), strict=True
    ):
        assert name == other
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)
    assert "logprobs" in reference["loss_fn_outputs"][0]
    assert result["loss_fn_outputs"] == [{}]


def test_zero_advantage_sequence_skips_model_but_keeps_normalization(model_path, monkeypatch):
    trainer = LocalTrainer(
        ModelSpec(base_model="tiny", base_path=model_path, rank=2), runtime=Runtime()
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("Zero-advantage sequence must not run the model")

    monkeypatch.setattr(trainer.model, "forward", unexpected)
    batch = Batch.model_validate(
        {
            "loss_fn": "ppo",
            "data": [
                {
                    "model_input": {"chunks": [{"tokens": [1, 4]}]},
                    "loss_fn_inputs": {
                        "target_tokens": {"dtype": "int64", "data": [4, 5]},
                        "weights": {"dtype": "float32", "data": [1.0, 1.0]},
                        "advantages": {"dtype": "float32", "data": [0.0, 0.0]},
                        "logprobs": {"dtype": "float32", "data": [-3.0, -3.0]},
                    },
                }
            ],
        }
    )
    result = trainer.forward(batch, True, return_outputs=False)
    assert result["loss"] == 0 and trainer.loss_tokens == 2 and trainer.sequences == 1


@pytest.mark.parametrize("train_unembed", [False, True])
def test_sdk_full_token_outputs_use_bounded_projection_with_identical_gradients(
    model_path, train_unembed
):
    spec = ModelSpec(base_model="tiny", base_path=model_path, rank=2, train_unembed=train_unembed)
    reference = LocalTrainer(spec, runtime=Runtime())
    chunked = LocalTrainer(spec, runtime=Runtime())
    for trainer in (reference, chunked):
        trainer.model.config.max_position_embeddings = 512
    ids = torch.tensor([[1, 4, 5, 6] * 65])
    targets = torch.tensor([4, 5, 6, 7] * 65)
    expected = (
        reference.model(input_ids=ids, use_cache=False)
        .logits[0]
        .float()
        .log_softmax(-1)
        .gather(-1, targets[:, None])
        .squeeze(-1)
    )
    (-expected.sum()).backward()
    projected = []
    handle = chunked.model.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: projected.append(args[0].shape[-2])
    )
    batch = Batch.model_validate(
        {
            "data": [
                {
                    "model_input": {"chunks": [{"tokens": ids[0].tolist()}]},
                    "loss_fn_inputs": {
                        "target_tokens": {"dtype": "int64", "data": targets.tolist()}
                    },
                }
            ]
        }
    )
    actual = chunked.forward(batch, backward=True)
    handle.remove()
    torch.testing.assert_close(
        torch.tensor(actual["loss_fn_outputs"][0]["logprobs"]["data"]), expected.detach()
    )
    assert max(projected) <= 128
    assert len(actual["loss_fn_outputs"][0]["logprobs"]["data"]) == len(targets)
    for (name, full), (other, bounded) in zip(
        reference.model.named_parameters(), chunked.model.named_parameters(), strict=True
    ):
        assert name == other
        if full.grad is not None:
            torch.testing.assert_close(full.grad, bounded.grad, atol=1e-6, rtol=1e-5)


def test_cached_sampling_matches_full_prefix_probabilities(model_path):
    trainer = LocalTrainer(ModelSpec(base_model="tiny", base_path=model_path, rank=2))
    trainer.model.config.get_text_config().eos_token_id = None
    lengths = []
    hook = trainer.model.register_forward_pre_hook(
        lambda module, args, kwargs: lengths.append(kwargs["input_ids"].shape[-1]),
        with_kwargs=True,
    )
    result = sample(trainer, [1, 4, 5], SamplingParams(max_tokens=8, temperature=0))["sequences"][0]
    hook.remove()
    assert len(result["tokens"]) == 8
    assert lengths == [3] + [1] * 7
    prefix = [1, 4, 5]
    with torch.no_grad():
        for token, actual in zip(result["tokens"], result["logprobs"], strict=True):
            logits = trainer.model(input_ids=torch.tensor([prefix]), use_cache=False).logits[0, -1]
            assert actual == pytest.approx(logits.log_softmax(-1)[token].item(), abs=1e-5)
            prefix.append(token)


def test_sampling_budget_uses_model_context_and_still_stops_at_eos(model_path):
    trainer = LocalTrainer(ModelSpec(base_model="tiny", base_path=model_path, rank=2))
    config = trainer.model.config.get_text_config()
    config.max_position_embeddings = 4096
    config.eos_token_id = 0
    with torch.no_grad():
        trainer.model.get_output_embeddings().weight.zero_()
    result = sample(trainer, [1, 4, 5], SamplingParams(max_tokens=2048, temperature=0))
    assert result["sequences"][0]["tokens"] == [0]
    assert result["sequences"][0]["stop_reason"] == "stop"
    with pytest.raises(ValueError, match="context limit"):
        sample(trainer, [1, 4, 5], SamplingParams(max_tokens=4096, temperature=0))
