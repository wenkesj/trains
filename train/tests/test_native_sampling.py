import json
from uuid import uuid4

import httpx
import pytest

from train.backend import LocalTrainer
from train.checkpoints import Checkpoint
from train.models import ModelSpec
from train.runtime import Runtime
from train.service import TrainingService
from train.serving import Serving


@pytest.fixture
def native_service(model_path, tmp_path, monkeypatch):
    calls, loaded = [], set()
    state = {"calls": calls, "mutate": lambda outputs: outputs}
    client = httpx.Client

    def handle(request):
        payload = json.loads(request.content) if request.content else None
        calls.append((request.url.path, payload))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": name} for name in ["tiny", *loaded]]})
        if request.url.path == "/load_lora_adapter":
            loaded.add(payload["lora_name"])
            return httpx.Response(200, json={"success": True})
        assert request.url.path == "/generate"
        assert payload["sampling_params"]["no_stop_trim"] is True
        assert payload["sampling_params"]["skip_special_tokens"] is False
        prompt = payload["input_ids"]
        tokens = [7, 2] if payload["sampling_params"]["max_new_tokens"] else []
        k = payload["top_logprobs_num"]
        outputs = [
            {
                "prompt_token_ids": prompt[:],
                "output_ids": tokens[:],
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [[-0.5, token, None] for token in tokens],
                    "input_token_logprobs": [
                        [None if i == 0 else -0.25, token, None] for i, token in enumerate(prompt)
                    ],
                    "output_top_logprobs": [
                        [[-0.5 - i, (token + i) % 32, None] for i in range(k)] for token in tokens
                    ],
                },
            }
            for _ in range(payload["sampling_params"]["n"])
        ]
        return httpx.Response(200, content=json.dumps(state["mutate"](outputs)))

    monkeypatch.setattr(
        "train.native_sampling.httpx.Client",
        lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs),
    )
    runtime = Runtime(
        sampling_backend="sglang",
        serving=Serving(backend="sglang", url="http://native.invalid", base_model="tiny"),
    )
    service = TrainingService({"tiny": model_path}, tmp_path / "native", runtime)
    service.root.mkdir()
    yield service, state
    service.executor.shutdown(wait=True)


def request(**changes):
    return {
        "model_path": "base://tiny",
        "prompt": {"chunks": [{"tokens": [1, 4, 5]}]},
        "sampling_params": {"max_tokens": 2, "temperature": 1, "stop": [2]},
        "num_samples": 2,
        "prompt_logprobs": True,
        "topk_sample_logprobs": 2,
        **changes,
    }


def test_native_sampling_preserves_tokens_scores_stops_and_sample_count(native_service):
    service, state = native_service
    result = service._generate(request())
    assert len(result["sequences"]) == 2
    for sequence in result["sequences"]:
        assert sequence["tokens"] == [7, 2]
        assert sequence["logprobs"] == sequence["sampling_logprobs"] == [-0.5, -0.5]
        assert sequence["stop_reason"] == "stop"
        assert sequence["topk_logprobs"] == [[(7, -0.5), (8, -1.5)], [(2, -0.5), (3, -1.5)]]
    assert result["prompt_logprobs"] == [None, -0.25, -0.25]
    payload = state["calls"][-1][1]
    assert payload["input_ids"] == [1, 4, 5]
    assert payload["sampling_params"]["stop_token_ids"] == [2]
    assert payload["sampling_params"]["top_k"] == -1
    assert service.sampler_engines == {}


def test_native_sampler_selects_the_exact_immutable_lora_checkpoint(native_service, model_path):
    service, state = native_service
    engine = LocalTrainer(ModelSpec(base_model="tiny", base_path=model_path, rank=2))
    checkpoint = Checkpoint.publish(
        service.root, "owner", "sampler", uuid4().hex, engine, True, False
    )
    result = service._generate(request(model_path=checkpoint.uri))
    assert result["sequences"][0]["tokens"] == [7, 2]
    load = next(payload for path, payload in state["calls"] if path == "/load_lora_adapter")
    assert state["calls"][-1][1]["lora_path"] == load["lora_name"]
    assert load["lora_path"] == str(checkpoint.directory)


@pytest.mark.parametrize("temperature", [0.5, 1.0])
def test_other_temperatures_and_targeted_scores_use_transformers(native_service, temperature):
    service, state = native_service
    body = request(
        sampling_params={"max_tokens": 2, "temperature": temperature},
        num_samples=1,
        topk_sample_logprobs=0,
        target_prompt_logprobs={"data": [6, 7], "shape": [2, 1]},
    )
    result = service._generate(body)
    assert result["target_prompt_logprobs"]["shape"] == [2, 1]
    assert len(result["sequences"][0]["tokens"]) <= 2
    assert state["calls"] == []
    assert "base://tiny" in service.sampler_engines


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda out: [dict(out[0], prompt_token_ids=[1]), out[1]], "input token IDs"),
        (lambda out: [dict(out[0], output_ids=[8, 2]), out[1]], "misaligned"),
        (lambda out: out[:1], "incomplete sequence batch"),
    ],
)
def test_native_replay_corruption_rejects_the_whole_batch(native_service, mutate, match):
    service, state = native_service
    state["mutate"] = mutate
    with pytest.raises(ValueError, match=match):
        service._generate(request())
    assert service.sampler_engines == {}


def test_invalid_requests_are_rejected_before_backend_work(native_service):
    service, state = native_service
    for body in [
        request(num_samples=17),
        request(prompt={"chunks": [{"tokens": [32]}]}),
        request(sampling_params={"max_tokens": 64}),
    ]:
        with pytest.raises(ValueError):
            service._generate(body)
    assert state["calls"] == []


def test_native_sampling_configuration_requires_its_backend():
    with pytest.raises(ValueError, match="configured SGLang"):
        Runtime(sampling_backend="sglang")


def test_greedy_and_zero_sample_requests_preserve_probability_contract(native_service):
    service, state = native_service
    result = service._generate(request(sampling_params={"max_tokens": 2, "temperature": 0}))
    assert result["sequences"][0]["logprobs"] == [-0.5, -0.5]
    assert result["sequences"][0]["sampling_logprobs"] == [0.0, 0.0]
    assert service._generate(request(num_samples=0)) == {
        "sequences": [],
        "prompt_logprobs": [None, -0.25, -0.25],
    }
    state["calls"].clear()
    assert service._generate(request(num_samples=0, prompt_logprobs=False)) == {
        "sequences": [],
        "prompt_logprobs": None,
    }
    assert state["calls"] == []
