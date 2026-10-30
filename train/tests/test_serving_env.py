from pathlib import Path

from train.runtime import Runtime


def test_inference_environment_configures_nested_runtime_and_auth(monkeypatch):
    settings = {
        "BACKEND": "sglang",
        "CHECKPOINT_MODE": "lora",
        "MODEL_PATH": "/models",
        "URL": "http://inference.invalid:30000",
        "API_KEY": "test-only",
        "TRAIN_ROOT": "/train",
        "BASE_MODEL": "LiquidAI/LFM3-Omni-2B",
        "SERVED_MODEL": "policy",
        "SHARED_GPU": "true",
        "TIMEOUT": "90",
        "MAX_OUTPUT_TOKENS": "2048",
    }
    for suffix, value in settings.items():
        monkeypatch.setenv("TRAIN_INFERENCE_" + suffix, value)
    serving = Runtime().serving
    assert serving.backend == "sglang"
    assert serving.checkpoint_mode == "lora"
    assert serving.model_path == Path("/models")
    assert serving.train_root == Path("/train")
    assert serving.base_model == settings["BASE_MODEL"]
    assert serving.adapter_model(serving.base_model, "adapter") == "policy:adapter"
    assert serving.endpoint("/generate") == settings["URL"] + "/generate"
    assert serving.headers == {"Authorization": "Bearer test-only"}
    assert serving.shared_gpu is True
    assert serving.timeout == 90
    assert serving.max_output_tokens == 2048
