import json
from types import SimpleNamespace

import httpx
import pytest

from train.deployments import Deployments
from train.serving import Serving


@pytest.mark.parametrize(
    ("backend", "release", "resume"),
    [
        ("vllm", "/sleep", "/wake_up"),
        ("sglang", "/release_memory_occupation", "/resume_memory_occupation"),
    ],
)
def test_shared_gpu_memory_is_restored_after_failed_training(monkeypatch, backend, release, resume):
    requests = []

    def reply(request):
        requests.append(request)
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(reply))
    monkeypatch.setattr("train.serving.httpx.Client", lambda **kwargs: client)
    serving = Serving(backend=backend, url="http://inference", shared_gpu=True)
    with pytest.raises(RuntimeError, match="training failed"), serving.suspended():
        raise RuntimeError("training failed")
    assert [request.url.path for request in requests] == [release, resume]
    if backend == "sglang":
        assert all(json.loads(request.content) == {"tags": ["kv_cache"]} for request in requests)
    else:
        assert requests[0].url.params["level"] == "1"


@pytest.mark.parametrize("already_loaded", [False, True])
def test_sglang_uses_qualified_adapter_model_after_native_loading(
    tmp_path, monkeypatch, already_loaded
):
    serving = Serving(backend="sglang", url="http://inference", served_model="liquid")
    manager = Deployments(tmp_path, {}, serving)
    monkeypatch.setattr(manager, "_adapter", lambda uri: ("adapter", "/weights/adapter"))
    monkeypatch.setattr(
        manager, "pin", lambda uri: (uri, SimpleNamespace(base_model="LiquidAI/LFM3-Omni-2B"))
    )
    requests = []

    def reply(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"data": [{"id": "liquid"}, *([{"id": "adapter"}] if already_loaded else [])]},
            )
        assert request.url.path == "/load_lora_adapter"
        assert json.loads(request.content) == {
            "lora_name": "adapter",
            "lora_path": "/weights/adapter",
        }
        return httpx.Response(200, json={"success": True})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        assert manager.ensure_loaded(client, "checkpoint") == "liquid:adapter"
    assert len(requests) == (1 if already_loaded else 2)


def test_sglang_worker_failure_cannot_publish_an_unloaded_adapter(tmp_path, monkeypatch):
    manager = Deployments(tmp_path, {}, Serving(backend="sglang", url="http://inference"))
    monkeypatch.setattr(manager, "_adapter", lambda uri: ("adapter", "/weights/adapter"))
    monkeypatch.setattr(
        manager, "pin", lambda uri: (uri, SimpleNamespace(base_model="LiquidAI/LFM3-Omni-2B"))
    )

    def reply(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(
            200, json={"success": False, "error_message": "unsupported adapter shape"}
        )

    with (
        httpx.Client(transport=httpx.MockTransport(reply)) as client,
        pytest.raises(ValueError, match="unsupported adapter shape"),
    ):
        manager.ensure_loaded(client, "checkpoint")
