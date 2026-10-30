import httpx
import pytest
import tinker
from tinker import types


def test_official_tinker_lora_train_restore_and_sample(training_server):
    service = tinker.ServiceClient(base_url=training_server, api_key="tml-local")
    assert service.get_server_capabilities().supported_models[0].model_name == "tiny"
    training = service.create_lora_training_client(base_model="tiny", rank=2, train_unembed=False)
    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints([1, 4, 5]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[4, 5, 6], dtype="int64", shape=[3])
            },
        )
    ]
    forward = training.forward_backward(data, "cross_entropy").result(timeout=15)
    assert forward.loss_fn_outputs[0]["logprobs"].shape == [3]
    assert len(forward.loss_fn_outputs[0]["logprobs"].data) == 3
    assert (
        training.optim_step(types.AdamParams(learning_rate=0.001))
        .result(timeout=15)
        .metrics["step"]
        == 1
    )
    state = training.save_state("checkpoint").result(timeout=15)
    training.load_state_with_optimizer(state.path).result(timeout=15)
    sampling = training.save_weights_and_get_sampling_client()
    result = sampling.sample(
        prompt=types.ModelInput.from_ints([1, 4]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=2, temperature=0),
        include_prompt_logprobs=True,
        topk_sample_logprobs=2,
    ).result(timeout=15)
    assert len(result.sequences) == 1
    assert result.prompt_logprobs[0] is None
    assert len(result.sequences[0].topk_logprobs[0]) == 2
    assert len(result.sequences[0].tokens) == len(result.sequences[0].logprobs)
    assert (
        len(sampling.compute_logprobs(types.ModelInput.from_ints([1, 4])).result(timeout=15)) == 2
    )
    service.close("success").result(timeout=15)


def test_sequence_reuse_rejects_changed_payload(training_server):
    with httpx.Client(base_url=training_server) as client:
        session = client.post("/api/v1/create_session", json={}).json()["session_id"]
        body = {
            "session_id": session,
            "model_seq_id": 0,
            "base_model": "tiny",
            "lora_config": {"rank": 2},
        }
        first = client.post("/api/v1/create_model", json=body)
        assert first.status_code == 200, first.text
        assert client.post("/api/v1/create_model", json=body).json() == first.json()
        body["lora_config"]["rank"] = 3
        assert client.post("/api/v1/create_model", json=body).status_code == 422


def test_tinker_spindle_rl_keeps_old_sampler_snapshot(training_server):
    service = tinker.ServiceClient(base_url=training_server, api_key="tml-local")
    trainer = service.create_lora_training_client("tiny", rank=2, train_unembed=False)
    old_sampler = trainer.save_weights_and_get_sampling_client()
    prompt = types.ModelInput.from_ints([1, 4])
    params = types.SamplingParams(max_tokens=2, temperature=0)
    before = old_sampler.sample(prompt, 1, params).result(timeout=15).sequences[0]
    completion = list(map(int, before.tokens))
    tokens = [1, 4, *completion]
    size = len(tokens) - 1
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(data=tokens[1:], dtype="int64", shape=[size]),
            "weights": types.TensorData(
                data=[0.0, *[1.0] * len(completion)], dtype="float32", shape=[size]
            ),
            "advantages": types.TensorData(
                data=[0.0, *[1.0] * len(completion)], dtype="float32", shape=[size]
            ),
            "logprobs": types.TensorData(
                data=[0.0, *map(float, before.logprobs)], dtype="float32", shape=[size]
            ),
        },
    )
    trainer.forward_backward([datum], "ppo").result(timeout=15)
    trainer.optim_step(types.AdamParams(learning_rate=0.01)).result(timeout=15)
    published = trainer.save_weights_for_sampler("updated").result(timeout=15)
    updated = service.create_sampling_client(model_path=published.path)
    after = updated.sample(prompt, 1, params).result(timeout=15).sequences[0]
    old_again = old_sampler.sample(prompt, 1, params).result(timeout=15).sequences[0]
    assert list(old_again.tokens) == list(before.tokens)
    assert list(old_again.logprobs) == pytest.approx(list(before.logprobs))
    assert list(after.logprobs) != list(before.logprobs)
    checkpoint = trainer.save_state("resume").result(timeout=15)
    restored = service.create_training_client_from_state_with_optimizer(checkpoint.path)
    restored.forward_backward([datum], "ppo").result(timeout=15)
    assert restored.optim_step(types.AdamParams()).result(timeout=15).metrics["step"] == 2
    service.close("success").result(timeout=15)


def test_interleaved_native_sessions_preserve_gradients_across_cache_eviction(training_server):
    service = tinker.ServiceClient(base_url=training_server, api_key="tml-local")
    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints([1, 4, 5]),
            loss_fn_inputs={
                "target_tokens": types.TensorData(data=[4, 5, 6], dtype="int64", shape=[3])
            },
        )
    ]
    first = service.create_lora_training_client("tiny", rank=2, train_unembed=False)
    first.forward_backward(data, "cross_entropy").result(timeout=15)
    second = service.create_lora_training_client("tiny", rank=2, train_unembed=False)
    second.forward_backward(data, "cross_entropy").result(timeout=15)
    params = types.AdamParams(learning_rate=0.01)
    assert first.optim_step(params).result(timeout=15).metrics["step"] == 1
    assert second.optim_step(params).result(timeout=15).metrics["step"] == 1
    left = first.forward(data, "cross_entropy").result(timeout=15)
    right = second.forward(data, "cross_entropy").result(timeout=15)
    assert left.loss_fn_outputs[0]["logprobs"].data == right.loss_fn_outputs[0]["logprobs"].data
    assert httpx.get(training_server + "/health").json()["loaded_trainers"] == 1
    service.close("success").result(timeout=15)


def test_training_queue_and_optimizer_retry(training_server, poll):
    with httpx.Client(base_url=training_server) as client:
        session = client.post("/api/v1/create_session", json={}).json()["session_id"]
        future = client.post(
            "/api/v1/create_model",
            json={
                "session_id": session,
                "model_seq_id": 0,
                "base_model": "tiny",
                "lora_config": {"rank": 2, "train_unembed": False},
            },
        ).json()
        model = poll(client, future["request_id"])["model_id"]
        batch = {
            "data": [
                {
                    "model_input": {"chunks": [{"tokens": [1, 4, 5]}]},
                    "loss_fn_inputs": {"target_tokens": {"data": [4, 5, 6], "dtype": "int64"}},
                }
            ],
            "loss_fn": "cross_entropy",
        }
        forward = {"model_id": model, "seq_id": 1, "forward_backward_input": batch}
        poll(client, client.post("/api/v1/forward_backward", json=forward).json()["request_id"])
        optimizer = {"model_id": model, "seq_id": 2, "adam_params": {"learning_rate": 0.01}}
        first = client.post("/api/v1/optim_step", json=optimizer).json()
        assert client.post("/api/v1/optim_step", json=optimizer).json() == first
        assert poll(client, first["request_id"])["metrics"]["step"] == 1
        changed = {**optimizer, "adam_params": {"learning_rate": 0.02}}
        assert client.post("/api/v1/optim_step", json=changed).status_code == 422
        pending = client.post("/api/v1/optim_step", json={**optimizer, "seq_id": 4}).json()
        assert (
            client.post(
                "/api/v1/retrieve_future", json={"request_id": pending["request_id"]}
            ).status_code
            == 408
        )
        client.post("/api/v1/forward_backward", json={**forward, "seq_id": 3})
        assert poll(client, pending["request_id"])["metrics"]["step"] == 2
        client.post(f"/api/v1/sessions/{session}/finish", json={})
        assert client.post("/api/v1/optim_step", json={**optimizer, "seq_id": 5}).status_code == 422
