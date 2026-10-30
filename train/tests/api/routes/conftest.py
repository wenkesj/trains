import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from trains.datasets import datum_dataset_from_datums


@pytest.fixture
def vendor_sdk(training_server, tmp_path):

    def check(provider):
        dataset = datum_dataset_from_datums(
            [
                {
                    "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 4, 5]}]},
                    "loss_fn_inputs": {
                        "target_tokens": {"data": [4, 5, 6], "dtype": "int64", "shape": [3]}
                    },
                }
            ]
        )
        path = tmp_path / "datum.json"
        path.write_text(dataset[0]["datum_json"])
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "contracts/clients.py"),
                provider,
                training_server,
                str(path),
            ],
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[4] / "trains")},
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        print(result.stdout)

    return check


def poll_future(client, identifier):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = client.post("/api/v1/retrieve_future", json={"request_id": identifier})
        if result.status_code == 200:
            assert "error" not in result.json(), result.text
            return result.json()
        assert result.status_code == 408, result.text
        time.sleep(0.01)
    raise AssertionError("Operation did not complete")


@pytest.fixture
def poll():
    return poll_future
