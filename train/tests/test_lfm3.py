import os

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from train.backend import LocalTrainer
from train.models import AdamParams, Batch, ModelSpec, SamplingParams
from train.runtime import Runtime
from train.sampling import sample


@pytest.fixture
def liquid_path(tmp_path):
    snapshot = os.environ.get("TRAIN_TEST_LFM3_SNAPSHOT")
    if not snapshot:
        pytest.skip("requires a pinned LiquidAI/LFM3-1.7B snapshot")
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=True)
    for name, value in {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_hidden_layers": 3,
        "layer_types": ["conv", "sliding_attention", "full_attention"],
        "vocab_size": 64,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "max_position_embeddings": 128,
        "sliding_window": 8,
    }.items():
        setattr(config, name, value)
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).float()
    path = tmp_path / "liquid"
    model.save_pretrained(path)
    return path


@pytest.mark.parametrize("objective", ["cross_entropy", "ppo", "importance_sampling"])
def test_lfm3_selected_projection_and_checkpointed_mlp_preserve_gradients(liquid_path, objective):
    spec = ModelSpec(
        base_model="LiquidAI/LFM3-1.7B", base_path=liquid_path, rank=2, train_unembed=False
    )
    plain = LocalTrainer(spec, runtime=Runtime(trust_remote_code=True, mlp_chunk_size=0))
    optimized = LocalTrainer(
        spec,
        runtime=Runtime(trust_remote_code=True, gradient_checkpointing=True, mlp_chunk_size=4),
    )
    datum = {
        "model_input": {"chunks": [{"tokens": list(range(1, 17))}]},
        "loss_fn_inputs": {
            "target_tokens": {"dtype": "int64", "data": list(range(2, 18))},
            "weights": {"dtype": "float32", "data": [0] * 8 + [1] * 8},
        },
    }
    if objective != "cross_entropy":
        datum["loss_fn_inputs"].update(
            logprobs={"dtype": "float32", "data": [-4.0] * 16},
            advantages={"dtype": "float32", "data": [0] * 8 + [0.5, -0.5] * 4},
        )
    batch = Batch.model_validate({"data": [datum], "loss_fn": objective})
    reference = plain.forward(batch, True)
    result = optimized.forward(batch, True, return_outputs=False)
    assert result["loss"] == pytest.approx(reference["loss"], abs=1e-5)
    for (name, left), (other, right) in zip(
        plain.model.named_parameters(), optimized.model.named_parameters(), strict=True
    ):
        assert name == other
        if left.grad is not None:
            assert right.grad is not None, name
            torch.testing.assert_close(left.grad, right.grad, atol=1e-5, rtol=1e-4)


def test_lfm3_training_restores_optimizer_and_requires_explicit_code_opt_in(liquid_path, tmp_path):
    spec = ModelSpec(
        base_model="LiquidAI/LFM3-1.7B", base_path=liquid_path, rank=4, train_unembed=False
    )
    with pytest.raises(ValueError, match="trust_remote_code"):
        Runtime().load(spec)
    trainer = LocalTrainer(spec, runtime=Runtime(trust_remote_code=True))
    batch = Batch.model_validate(
        {
            "data": [
                {
                    "model_input": {"chunks": [{"tokens": [1, 4, 5, 6]}]},
                    "loss_fn_inputs": {"target_tokens": {"dtype": "int64", "data": [4, 5, 6, 7]}},
                }
            ],
        }
    )
    before = trainer.forward(batch, False)["loss"]
    for _ in range(3):
        trainer.forward(batch, True, return_outputs=False)
        trainer.optim_step(AdamParams(learning_rate=0.01), normalization="num_loss_tokens")
    after = trainer.forward(batch, False)["loss"]
    assert after < before
    trainer.save(tmp_path / "checkpoint", sampler=False)
    restored = LocalTrainer(spec, runtime=Runtime(trust_remote_code=True))
    restored.load(tmp_path / "checkpoint", optimizer=True)
    assert restored.step == 3
    assert restored.forward(batch, False)["loss"] == pytest.approx(after)


def test_lfm3_cached_convolution_and_attention_match_full_prefix_sampling(liquid_path):
    trainer = LocalTrainer(
        ModelSpec(base_model="LiquidAI/LFM3-1.7B", base_path=liquid_path, rank=2),
        runtime=Runtime(trust_remote_code=True),
    )
    trainer.model.config.eos_token_id = None
    result = sample(trainer, [1, 4, 5], SamplingParams(max_tokens=8, temperature=0))["sequences"][0]
    prefix = [1, 4, 5]
    with torch.no_grad():
        for token, actual in zip(result["tokens"], result["logprobs"], strict=True):
            output = trainer.model(input_ids=torch.tensor([prefix]), use_cache=False)
            expected = output.logits[0, -1].log_softmax(-1)[token].item()
            assert actual == pytest.approx(expected, abs=1e-5)
            prefix.append(token)
