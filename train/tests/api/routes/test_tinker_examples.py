import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import tinker
from recipes.rl import ppo_update
from tinker import types
from trains.datasets.serialization import datum_mapping
from trains.datum.adapters.spindle import SpindleDatumAdapter
from trains.datum.adapters.tinker import TinkerDatumAdapter


@pytest.mark.parametrize("example", ["harbor", "inspect"])
@pytest.mark.parametrize("provider", ["tinker", "spindle"])
def test_native_train_publish_and_recollect(example, provider, training_server, tmp_path):
    service = tinker.ServiceClient(base_url=training_server, api_key="tml-local")
    try:
        client = service.create_lora_training_client(base_model="tiny", rank=2, train_unembed=False)
        adapter = TinkerDatumAdapter() if provider == "tinker" else SpindleDatumAdapter()
        rows = [
            {
                "group_id": "task",
                "trace_id": str(index),
                "reward": index,
                "model_id": "behavior",
                "metadata_json": '{"sampling":{"temperature":1}}',
                "datum_json": json.dumps(datum_mapping(adapter.from_tokens([1], tokens))),
            }
            for index, tokens in enumerate(([4, 5], [6, 7]))
        ]
        ppo_update(rows, client, types, behavior_model="behavior", learning_rate=0.001)
        state = client.save_state("example-state").result()
        receipt = tmp_path / "checkpoint.json"
        receipt.write_text(
            json.dumps(
                {
                    "state": state.path,
                    "base_model": "tiny",
                    "training_url": training_server,
                    "provider": provider,
                }
            )
        )
        deployment = tmp_path / "deployment.json"
        script = Path(__file__).resolve().parents[4] / "examples" / example / "run.py"
        subprocess.run(
            [
                sys.executable,
                str(script),
                "deploy",
                str(receipt),
                "--inference-url",
                training_server + "/oai/api/v1",
                "--output",
                str(deployment),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "TINKER_API_KEY": "tml-local"},
        )
        published = json.loads(deployment.read_text())
        assert published["model"].startswith("tinker://")
        assert published["provider"] == provider
        response = httpx.post(
            published["inference_url"] + "/chat/completions",
            timeout=30,
            json={
                "model": published["model"],
                "messages": [{"role": "user", "content": "after update"}],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "ok"
    finally:
        service.close("success").result()


@pytest.mark.parametrize("example", ["harbor", "inspect"])
def test_initial_sampler_publication(example, training_server, tmp_path):
    output = tmp_path / "initial.json"
    script = Path(__file__).resolve().parents[4] / "examples" / example / "run.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "deploy",
            "--base-model",
            "tiny",
            "--rank",
            "2",
            "--training-url",
            training_server,
            "--inference-url",
            training_server + "/oai/api/v1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "TINKER_API_KEY": "tml-local"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    initial = json.loads(output.read_text())
    response = httpx.post(
        initial["inference_url"] + "/chat/completions",
        timeout=30,
        json={"model": initial["model"], "messages": []},
    )
    assert response.status_code == 200, response.text
