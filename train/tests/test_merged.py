import json
from pathlib import Path

import httpx
import pytest
import torch
from transformers import AutoModelForCausalLM

from train.backend import LocalTrainer
from train.merged import MergedWeights
from train.models import ModelSpec
from train.runtime import Runtime
from train.serving import Serving


@pytest.fixture
def merged_case(model_path, tmp_path):
    serving = Serving(
        backend="sglang", checkpoint_mode="merged", url="http://sglang", served_model="tiny"
    )
    runtime = Runtime(serving=serving)
    spec = ModelSpec(base_model="tiny", base_path=model_path, rank=2, train_unembed=False)
    trainer = LocalTrainer(spec, runtime=runtime)
    with torch.no_grad():
        for name, parameter in trainer.model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.1)
    checkpoint = tmp_path / "sampler"
    trainer.save(checkpoint, sampler=True)
    manager = MergedWeights(tmp_path / "rpc", runtime)
    state = {"version": "default", "model": "tiny", "posts": [], "failure": False, "activate": True}

    def reply(request):
        if request.method == "GET":
            assert request.url.path == "/get_model_info"
            return httpx.Response(
                200, json={"weight_version": state["version"], "served_model_name": state["model"]}
            )
        assert request.url.path == "/update_weights_from_disk"
        payload = json.loads(request.content)
        state["posts"].append(payload)
        assert payload["flush_cache"] is True
        assert Path(payload["model_path"]).is_dir()
        if state["failure"]:
            return httpx.Response(200, json={"success": False, "message": "rejected"})
        if state["activate"]:
            state["version"] = payload["weight_version"]
            state["model"] = payload["model_path"]
        return httpx.Response(200, json={"success": True})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        yield manager, client, spec, checkpoint, trainer, state


def test_merged_weights_match_the_actual_adapter_and_restore_original_base(merged_case):
    manager, client, spec, checkpoint, trainer, state = merged_case
    ids = torch.tensor([[1, 3, 5]])
    with torch.no_grad():
        expected = trainer.model(input_ids=ids).logits
    assert manager.select(client, spec, checkpoint) == str(manager.directory / "merged")
    loaded = AutoModelForCausalLM.from_pretrained(state["posts"][-1]["model_path"])
    with torch.no_grad():
        torch.testing.assert_close(loaded(input_ids=ids).logits, expected, atol=1e-5, rtol=1e-5)
    manager.select(client, spec, checkpoint)
    assert len(state["posts"]) == 1
    manager.select(client, spec, None)
    assert state["posts"][-1]["model_path"] == str(spec.base_path)
    manager.select(client, spec, checkpoint)
    assert len(state["posts"]) == 3
    assert len(list(manager.directory.rglob("*.safetensors"))) == 1
    assert (checkpoint / "adapter_model.safetensors").is_file()


def test_backend_restart_reloads_cached_export_without_remerging(merged_case, monkeypatch):
    manager, client, spec, checkpoint, _, state = merged_case
    manager.select(client, spec, checkpoint)
    state["version"] = "default"
    restarted = MergedWeights(manager.root, manager.runtime)

    def unexpected_export(*args):
        pytest.fail("A backend restart should reuse the existing immutable checkpoint export")

    monkeypatch.setattr(restarted, "_export", unexpected_export)
    restarted.select(client, spec, checkpoint)
    assert len(state["posts"]) == 2


def test_failed_update_invalidates_previous_active_marker(merged_case):
    manager, client, spec, checkpoint, _, state = merged_case
    manager.select(client, spec, None)
    state["failure"] = True
    with pytest.raises(ValueError, match="rejected"):
        manager.select(client, spec, checkpoint)
    state["failure"] = False
    manager.select(client, spec, None)
    assert len(state["posts"]) == 3
    assert state["posts"][-1]["model_path"] == str(spec.base_path)


def test_http_success_without_the_requested_version_cannot_activate(merged_case):
    manager, client, spec, checkpoint, _, state = merged_case
    state["activate"] = False
    with pytest.raises(ValueError, match="did not activate"):
        manager.select(client, spec, checkpoint)
    assert json.loads((manager.directory / "state.json").read_text())["active"] is None


def test_merged_export_normalizes_inactive_greedy_sampling_options(merged_case):
    manager, client, spec, checkpoint, _, state = merged_case
    path = spec.base_path / "generation_config.json"
    config = json.loads(path.read_text())
    config.update(do_sample=False, temperature=0, top_k=1)
    path.write_text(json.dumps(config))
    manager.select(client, spec, checkpoint)
    exported = json.loads(
        (Path(state["posts"][-1]["model_path"]) / "generation_config.json").read_text()
    )
    assert exported.get("do_sample", False) is False
    assert exported.get("temperature", 1) == 1


def test_merged_mode_requires_native_sglang_weight_api():
    with pytest.raises(ValueError, match="requires the SGLang"):
        Serving(backend="vllm", checkpoint_mode="merged")
