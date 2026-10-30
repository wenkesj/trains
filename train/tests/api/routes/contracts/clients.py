"""Run native training and management clients together in the test environment."""

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
from baseten import loops
from fireworks import Fireworks
from fireworks.training import sdk
from fireworks.training.sdk.sampling import DeploymentSampler
from trains.datasets.serialization import datum_mapping
from trains.datum.adapters.baseten import BasetenDatumAdapter
from trains.datum.adapters.fireworks import FireworksDatumAdapter


def baseten(url):
    service = loops.ServiceClient(base_url=url, api_key="local")
    training = service.create_lora_training_client(base_model="tiny", rank=2)
    datum = native_datum(loops)
    result = training.forward_backward([datum]).result(timeout=15)
    assert len(result.loss_fn_outputs[0]["logprobs"].data) == 3
    assert training.optim_step(loops.AdamParams()).result(timeout=15).metrics["step"] == 1
    checkpoint = training.save_state("checkpoint").result(timeout=15)
    training.load_state_with_optimizer(checkpoint.path).result(timeout=15)
    sampling = training.save_weights_and_get_sampling_client("sampler")
    response = sampling.sample(
        loops.ModelInput(chunks=[loops.EncodedTextChunk(tokens=[1, 4])]),
        sampling_params=loops.SamplingParams(max_tokens=2, temperature=0),
        include_prompt_logprobs=True,
    )
    assert len(response.sequences[0].tokens) == len(response.sequences[0].logprobs)
    assert response.prompt_logprobs[0] is None
    # Reusing the named publication must not collide with its prior snapshot.
    republished = training.save_weights_and_get_sampling_client("sampler")
    republished.close()
    sampling.close()
    restored = service.create_training_client_from_state_with_optimizer(checkpoint.path)
    restored.forward_backward([datum]).result(timeout=15)
    assert restored.optim_step(loops.AdamParams()).result(timeout=15).metrics["step"] == 2
    snapshot = training.save_weights_for_sampler("standalone").result(timeout=15)
    standalone = service.create_sampling_client(model_path=snapshot.path)
    assert standalone.sample(
        loops.ModelInput(chunks=[loops.EncodedTextChunk(tokens=[1, 4])]),
        sampling_params=loops.SamplingParams(max_tokens=1),
    ).sequences
    standalone.close()
    with httpx.Client(base_url=url, timeout=30) as http:
        rows = http.get("/v1/loops/checkpoints").json()["checkpoints"]
        row = next(row for row in rows if row["checkpoint_id"] == "standalone")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("deploy_baseten.py")),
                url,
                row["id"],
            ],
            check=True,
        )
    selected = next(
        item
        for item in service.list_checkpoints(training.run_id)
        if item.checkpoint_id == "standalone" and item.target == "sampler"
    )
    receipt = {
        "state": checkpoint.path,
        "training_url": url,
        "checkpoint_id": selected.id,
        "sampler_name": selected.checkpoint_id,
        "base_model": "tiny",
        "sampler": snapshot.path,
    }
    for example in ("harvey_lab",):
        deploy_example(
            example,
            receipt,
            url,
            [
                "--local",
                "--model-name",
                example,
                "--instance-type",
                "local",
                "--hf-secret",
                "unused-local",
            ],
        )
    restored.close()
    training.close()
    print(
        "Baseten Loops: training, restore, sampling, native example deployments and inference passed"
    )


def fireworks(url):
    service = sdk.ServiceClient(base_url=url, api_key="tml-local")
    training = service.create_lora_training_client(base_model="tiny", rank=2, train_unembed=False)
    datum = native_datum(sdk)
    result = training.forward_backward([datum], "cross_entropy").result(timeout=15)
    assert len(result.loss_fn_outputs[0]["logprobs"].data) == 3
    training.optim_step(sdk.AdamParams(learning_rate=0.001)).result(timeout=15)
    checkpoint = training.save_state("checkpoint").result(timeout=15)
    training.load_state_with_optimizer(checkpoint.path).result(timeout=15)
    published = training.save_weights_for_sampler("sampler").result(timeout=15)
    deployment = DeploymentSampler(inference_url=url, model=published.path, api_key="local")
    sampling = service.create_sampling_client(deployment_sampler=deployment)
    response = sampling.sample(
        sdk.ModelInput.from_ints([1, 4]),
        1,
        sdk.SamplingParams(max_tokens=2, temperature=0),
        include_prompt_logprobs=True,
    ).result(timeout=15)
    assert len(response.sequences[0].tokens) == len(response.sequences[0].logprobs)
    assert response.prompt_logprobs[0] is None
    sampling.close()
    control = sdk.FireworksClient(api_key="local", base_url=url)
    job = published.path.split("/")[1].removeprefix("run-")
    row = next(row for row in control.list_checkpoints(job) if row["promotable"])
    promoted = control.promote_checkpoint(
        name=row["name"], output_model_id="sdk-policy", base_model="tiny"
    )
    assert promoted["state"] == "READY"
    with Fireworks(api_key="local", base_url=url) as cloud:
        deployment = cloud.deployments.create(
            account_id="local", deployment_id="sdk-serving", base_model=promoted["name"]
        )
        assert deployment.state == "READY"
        with httpx.Client(base_url=url, timeout=30) as http:
            response = http.post(
                "/inference/v1/chat/completions", json={"model": promoted["name"], "messages": []}
            )
            assert response.status_code == 200, response.text
        cloud.deployments.delete("sdk-serving", account_id="local")
    for example in ("ultrachat",):
        deploy_example(
            example,
            {"sampler": published.path, "base_model": "tiny"},
            url,
            ["--account-id", "local", "--model-id", example, "--deployment-id", example],
        )
    service.close()
    print("Firetitan: training, restore, native example deployments and inference passed")


def deploy_example(example, receipt, url, options):
    directory = Path(sys.argv[3]).parent
    checkpoint = directory / f"{example}-checkpoint.json"
    output = directory / f"{example}-deployment.json"
    checkpoint.write_text(json.dumps(receipt))
    script = Path(__file__).resolve().parents[5] / "examples" / example / "run.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "deploy",
            str(checkpoint),
            "--control-url",
            url,
            "--output",
            str(output),
            *options,
        ],
        check=True,
        timeout=60,
        env={**os.environ, "BASETEN_API_KEY": "local", "FIREWORKS_API_KEY": "local"},
    )
    deployment = json.loads(output.read_text())
    response = httpx.post(
        deployment["inference_url"] + "/chat/completions",
        timeout=30,
        json={
            "model": deployment["model"],
            "messages": [{"role": "user", "content": "after update"}],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "ok"


def native_datum(types):
    with open(sys.argv[3]) as stream:
        value = json.load(stream)
    adapter = BasetenDatumAdapter() if types is loops else FireworksDatumAdapter()
    tokens = value["model_input"]["chunks"][0]["tokens"]
    native = adapter.from_tokens(
        tokens[:1], [*tokens[1:], value["loss_fn_inputs"]["target_tokens"]["data"][-1]]
    )
    assert datum_mapping(native)["model_input"]["chunks"][0]["tokens"] == tokens
    return native


if __name__ == "__main__":
    {"baseten": baseten, "fireworks": fireworks}[sys.argv[1]](sys.argv[2])
