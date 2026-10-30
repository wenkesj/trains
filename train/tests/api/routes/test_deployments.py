import asyncio
import fcntl
import json
from pathlib import Path

import httpx
import pytest
import tinker
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from train.api.api import create_app
from train.api.routes import inference
from train.api.routes.inference import _rewrite_model, proxy
from train.backend import LocalTrainer
from train.checkpoints import Checkpoint
from train.models import ModelSpec
from train.runtime import Runtime
from train.serving import Serving


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_sglang_preserves_full_prompt_and_bounds_only_output_to_remaining_context(
    publication, monkeypatch, stream, field
):
    client, record, _, app = publication
    app.state.training.runtime.serving = Serving(
        backend="sglang", url="http://backend", max_output_tokens=4096
    )
    monkeypatch.setattr(inference, "_prepare", lambda *args: "native-policy")
    messages = [{"role": "user", "content": "The entire captured document stays intact."}]
    calls = []

    def backend(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/tokenize":
            assert body["messages"] == messages and body["model"] == "native-policy"
            assert not {field, "stream", "stream_options"}.intersection(body)
            return httpx.Response(200, json={"count": 30085, "max_model_len": 32768})
        assert request.url.path == "/v1/chat/completions"
        assert body["messages"] == messages and body[field] == 2683
        if stream:
            return httpx.Response(
                200,
                content='data: {"model":"native-policy","choices":[]}\n\ndata: [DONE]\n\n',
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json={"model": "native-policy", "choices": []})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        inference.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(backend), **kwargs),
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": record.uri,
            "messages": messages,
            field: 128000,
            "stream": stream,
            **({"stream_options": {"include_usage": True}} if stream else {}),
        },
    )
    assert response.status_code == 200, response.text
    assert record.uri in response.text
    assert [path for path, _ in calls] == ["/tokenize", "/v1/chat/completions"]


def test_context_exhausted_input_is_rejected_without_truncation(publication, monkeypatch):
    client, record, _, app = publication
    app.state.training.runtime.serving = Serving(
        backend="sglang", url="http://backend", max_output_tokens=4096
    )
    monkeypatch.setattr(inference, "_prepare", lambda *args: "native-policy")
    messages = [{"role": "user", "content": "Full input exceeds context."}]

    def backend(request):
        body = json.loads(request.content)
        assert body["messages"] == messages
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 33000, "max_model_len": 32768})
        assert body["max_tokens"] == 4096
        return httpx.Response(400, json={"error": "Input exceeds context"})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        inference.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(backend), **kwargs),
    )
    response = client.post("/v1/chat/completions", json={"model": record.uri, "messages": messages})
    assert response.status_code == 400 and response.json()["error"] == "Input exceeds context"


@pytest.mark.parametrize("messages", [None, "invalid", ["invalid"]])
def test_context_budget_preserves_backend_validation_of_invalid_messages(
    publication, monkeypatch, messages
):
    client, record, _, app = publication
    app.state.training.runtime.serving = Serving(
        backend="sglang", url="http://backend", max_output_tokens=4096
    )
    monkeypatch.setattr(inference, "_prepare", lambda *args: "native-policy")

    def backend(request):
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content)["messages"] == messages
        return httpx.Response(400, json={"error": "Invalid messages"})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        inference.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(backend), **kwargs),
    )
    response = client.post("/v1/chat/completions", json={"model": record.uri, "messages": messages})
    assert response.status_code == 400 and response.json()["error"] == "Invalid messages"


@pytest.fixture
def publication(model_path, tmp_path, vllm):
    root = tmp_path / "rpc"
    engine = LocalTrainer(
        ModelSpec(base_model="tiny", base_path=model_path, rank=2, train_unembed=False)
    )
    record = Checkpoint.publish(root, "a" * 32, "policy", "first", engine, True, False)
    runtime = Runtime(
        serving=Serving(url=vllm[0], api_key="server-key", train_root=Path("/mounted"))
    )
    app = create_app({"tiny": model_path}, root, runtime)
    with TestClient(app) as client:
        yield client, record, engine, app


def test_local_publication_pins_snapshot_and_survives_restart(publication, vllm):
    client, record, engine, app = publication
    body = {"checkpoint": record.uri, "model": "my-policy"}
    result = client.post("/v1/deployments", json=body)
    assert result.status_code == 200, result.text
    assert result.json()["base_url"] == "http://testserver/v1"
    assert client.post("/v1/deployments", json=body).status_code == 200
    calls = vllm[1]["calls"]
    loads = [body for path, body, _ in calls if path == "/v1/load_lora_adapter"]
    assert len(loads) == 1
    assert loads[0]["lora_path"] == str(
        Path("/mounted") / record.directory.relative_to(app.state.training.root)
    )
    assert all(auth == "Bearer server-key" for _, _, auth in calls)
    Checkpoint.publish(app.state.training.root, "a" * 32, "policy", "second", engine, True, True)
    assert client.post("/v1/deployments", json=body).status_code == 422
    messages = [{"role": "user", "content": "hi"}]
    response = client.post(
        "/v1/chat/completions", json={"model": "my-policy", "messages": messages}
    )
    assert response.status_code == 200
    assert response.json()["model"] == "my-policy"
    assert calls[-1][1]["model"] == loads[0]["lora_name"]
    vllm[1]["loaded"].clear()
    restarted = create_app(
        app.state.training.models, app.state.training.root, app.state.training.runtime
    )
    with TestClient(restarted) as other:
        response = other.post(
            "/v1/chat/completions", json={"model": "my-policy", "messages": messages}
        )
        assert response.status_code == 200
        assert calls[-1][1]["model"] == loads[0]["lora_name"]
        assert other.delete("/v1/deployments/my-policy").status_code == 200
        assert (
            other.post(
                "/v1/chat/completions", json={"model": "my-policy", "messages": messages}
            ).status_code
            == 404
        )
    assert not vllm[1]["loaded"]
    assert record.directory.exists()


def test_direct_tinker_spindle_checkpoint_chat_and_stream(publication, vllm):
    client, record, _, _ = publication
    for prefix in ("/v1", "/oai/api/v1"):
        with client.stream(
            "POST",
            prefix + "/chat/completions",
            json={
                "model": record.uri,
                "messages": [],
                "stream": True,
                "tools": [{"type": "function"}],
            },
        ) as response:
            assert response.status_code == 200
            assert "data: [DONE]" in response.read().decode()
        assert vllm[1]["calls"][-1][1]["tools"] == [{"type": "function"}]
    assert (
        client.post("/v1/responses", json={"model": record.uri, "input": "hi"}).status_code == 200
    )
    assert vllm[1]["calls"][-1][0] == "/v1/responses"


def test_baseten_deploy_multiple_checkpoints_and_delete(publication, vllm):
    client, record, engine, app = publication
    second = Checkpoint.publish(
        app.state.training.root, "a" * 32, "other", "second", engine, True, False
    )
    result = client.post(
        "/v1/loops/checkpoints/deploy",
        json={
            "checkpoint_ids": [record.directory.name, second.directory.name],
            "model_name": "policy",
            "instance_type_id": "local",
            "hf_secret_name": "unused-local",
        },
    )
    assert result.status_code == 200, result.text
    model, deployment = result.json()["model_id"], result.json()["deployment_id"]
    assert client.get(f"/v1/models/{model}/deployments/{deployment}").json()["status"] == "ACTIVE"
    for name in ("policy", "other"):
        response = client.post(
            f"/deployment/{deployment}/sync/v1/chat/completions",
            json={"model": name, "messages": []},
        )
        assert response.status_code == 200
    assert (
        client.post(
            f"/deployment/{deployment}/sync/v1/chat/completions",
            json={"model": "not-attached", "messages": []},
        ).status_code
        == 404
    )
    assert client.delete(f"/v1/models/{model}").status_code == 200
    assert not vllm[1]["loaded"]


def test_fireworks_promotion_deploy_and_multi_lora(publication):
    client, record, _, _ = publication
    checkpoints = client.get(f"/v1/accounts/local/rlorTrainerJobs/{'a' * 32}/checkpoints").json()[
        "checkpoints"
    ]
    assert checkpoints[0]["promotable"] is True
    model = "accounts/local/models/policy"
    result = client.post(
        f"/v1/accounts/local/checkpoints/{record.directory.name}:promote",
        json={
            "trainer_job_id": f"accounts/local/rlorTrainerJobs/{'a' * 32}",
            "output_model": model,
            "base_model": "tiny",
            "async_promotion": True,
        },
    )
    assert result.status_code == 200, result.text
    assert (
        client.post(
            "/inference/v1/chat/completions", json={"model": model, "messages": []}
        ).status_code
        == 404
    )
    for name, base, addons in (("merged", model, False), ("shared", "tiny", True)):
        result = client.post(
            f"/v1/accounts/local/deployments?deploymentId={name}",
            json={"baseModel": base, "enableAddons": addons},
        )
        assert result.status_code == 200, result.text
        deployment = result.json()["name"]
        if addons:
            assert (
                client.post(
                    "/v1/accounts/local/deployedModels",
                    json={"model": model, "deployment": deployment},
                ).status_code
                == 200
            )
        assert client.get(f"/v1/{deployment}").json()["state"] == "READY"
        assert (
            client.post(
                "/inference/v1/chat/completions",
                json={"model": f"{model}#{deployment}", "messages": []},
            ).status_code
            == 200
        )
        assert client.delete(f"/v1/{deployment}").status_code == 200


def test_failed_publication_is_not_ready_and_validation(publication, vllm):
    client, record, engine, app = publication
    vllm[1]["fail_load"] = True
    assert (
        client.post(
            "/v1/deployments", json={"checkpoint": record.uri, "model": "policy"}
        ).status_code
        == 502
    )
    assert client.get("/v1/deployments/policy").status_code == 404
    vllm[1]["fail_load"] = False
    assert (
        client.post(
            "/v1/deployments", json={"checkpoint": record.uri, "model": "policy"}
        ).status_code
        == 200
    )
    vllm[1]["fail_inference"] = True
    response = client.post("/v1/chat/completions", json={"model": "policy", "messages": []})
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "invalid request"
    with pytest.raises(ValueError, match="immutable snapshot"):
        Checkpoint.publish(
            app.state.training.root, "a" * 32, record.directory.name, "shadow", engine, True, True
        )
    state = Checkpoint.publish(
        app.state.training.root, "a" * 32, "state", "state", engine, False, False
    )
    for uri in (state.uri, "tinker://../../etc/passwd", "/tmp/adapter"):
        assert (
            client.post("/v1/deployments", json={"checkpoint": uri, "model": "invalid"}).status_code
            == 422
        )
    app.state.training.runtime.serving.base_model = "wrong"
    assert (
        client.post("/v1/chat/completions", json={"model": "policy", "messages": []}).status_code
        == 422
    )


def test_shared_gpu_wakes_on_training_failure(model_path, tmp_path, vllm):
    root = tmp_path / "rpc"
    root.mkdir()
    runtime = Runtime(serving=Serving(url=vllm[0], shared_gpu=True))
    with pytest.raises(RuntimeError), runtime.exclusive(root, lambda: None):
        with (root / ".lock").open("a") as handle, pytest.raises(BlockingIOError):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise RuntimeError("training failed")
    assert [path for path, _, _ in vllm[1]["calls"]] == ["/sleep?level=1", "/wake_up"]
    with (root / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_native_tinker_saved_checkpoint_is_callable(training_server):
    service = tinker.ServiceClient(base_url=training_server, api_key="tml-local")
    try:
        training = service.create_lora_training_client(
            base_model="tiny", rank=2, train_unembed=False
        )
        path = training.save_weights_for_sampler("chat").result().path
        response = httpx.post(
            training_server + "/oai/api/v1/chat/completions", json={"model": path, "messages": []}
        )
        assert response.status_code == 200, response.text
    finally:
        service.close("success").result()


def test_stream_disconnect_before_body_releases_gpu_lock(publication):
    _, record, _, app = publication

    async def check():
        response = await proxy(
            {"model": record.uri, "messages": [], "stream": True},
            "chat/completions",
            app.state.training,
        )
        with (app.state.training.root / ".lock").open("a") as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

            async def send(_message):
                raise OSError("client disconnected")

            async def receive():
                return {"type": "http.disconnect"}

            with pytest.raises(ClientDisconnect):
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    asyncio.run(check())


def test_anthropic_messages_routes_resolve_the_published_sampler(publication, vllm):
    client, record, _, _ = publication
    for path in ("/v1/messages", "/v1/messages/count_tokens"):
        body = {
            "model": record.uri,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
        }
        response = client.post(path, json=body)
        assert response.status_code == 200, response.text
        assert vllm[1]["calls"][-1][0] == path
        assert vllm[1]["calls"][-1][1]["messages"] == body["messages"]
        assert vllm[1]["calls"][-1][1]["model"] in vllm[1]["loaded"]
        assert response.json()["model"] == record.uri


def test_anthropic_stream_model_uses_the_public_sampler_identity():
    event = {"type": "message_start", "message": {"model": "internal-lora", "content": []}}
    actual = json.loads(_rewrite_model(json.dumps(event).encode(), "tinker://public/model"))
    assert actual["message"]["model"] == "tinker://public/model"
    assert actual["type"] == "message_start"


def test_baseten_responses_pin_the_deployment_and_reject_conflicting_selectors(publication, vllm):
    client, record, engine, app = publication
    second = Checkpoint.publish(
        app.state.training.root, "b" * 32, "policy", "second", engine, True, False
    )
    deployments = []
    for checkpoint in [record, second]:
        response = client.post(
            "/v1/loops/checkpoints/deploy",
            json={
                "checkpoint_ids": [checkpoint.directory.name],
                "model_name": "shared-alias",
                "instance_type_id": "local",
                "hf_secret_name": "unused-local",
            },
        )
        assert response.status_code == 200, response.text
        deployments.append(response.json()["deployment_id"])
    loaded_models = []
    for deployment in deployments:
        response = client.post(
            f"/deployment/{deployment}/sync/v1/responses",
            json={
                "model": "policy",
                "input": "Judge this answer",
                "max_output_tokens": 64,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["model"] == "policy"
        path, body, _ = vllm[1]["calls"][-1]
        assert path == "/v1/responses" and body["input"] == "Judge this answer"
        loaded_models.append(body["model"])
    assert loaded_models[0] != loaded_models[1]
    for path, body in [("responses", {"input": "judge"}), ("chat/completions", {"messages": []})]:
        before = len(vllm[1]["calls"])
        response = client.post(
            f"/deployment/{deployments[0]}/sync/v1/{path}",
            json={
                **body,
                "model": "policy#" + deployments[1],
            },
        )
        assert response.status_code == 422 and "does not match" in response.text
        assert len(vllm[1]["calls"]) == before
