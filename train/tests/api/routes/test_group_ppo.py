import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import tinker
from baseten import loops
from fireworks.training import sdk as fireworks
from recipes.rl import ppo_update
from tinker import types
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from trains.datasets import chat_dataset_from_records
from trains.datasets.serialization import datum_mapping
from trains.datum.adapters.baseten import BasetenDatumAdapter
from trains.datum.adapters.fireworks import FireworksDatumAdapter
from trains.datum.adapters.spindle import SpindleDatumAdapter
from trains.datum.adapters.tinker import TinkerDatumAdapter
from transformers import PreTrainedTokenizerFast


@pytest.mark.parametrize("provider", ["tinker", "spindle", "baseten", "fireworks"])
def test_common_group_ppo_updates_native_policy(provider, training_server):
    sdk = loops if provider == "baseten" else fireworks if provider == "fireworks" else types
    service = (
        loops.ServiceClient(base_url=training_server, api_key="local")
        if provider == "baseten"
        else fireworks.ServiceClient(base_url=training_server, api_key="tml-local")
        if provider == "fireworks"
        else tinker.ServiceClient(base_url=training_server, api_key="tml-local")
    )
    adapter = {
        "tinker": TinkerDatumAdapter,
        "spindle": SpindleDatumAdapter,
        "baseten": BasetenDatumAdapter,
        "fireworks": FireworksDatumAdapter,
    }[provider]()
    original = None
    restored = None
    try:
        kwargs = {"base_model": "tiny", "rank": 2}
        if provider != "baseten":
            kwargs["train_unembed"] = False
        original = service.create_lora_training_client(**kwargs)
        state = original.save_state("behavior").result()
        if provider == "fireworks":
            restored = original
            restored.load_state_with_optimizer(state.path).result()
        else:
            restored = service.create_training_client_from_state_with_optimizer(state.path)
        native = [adapter.from_tokens([1], tokens) for tokens in ([3, 4], [5, 6])]
        rows = [
            {
                "group_id": "task",
                "trace_id": str(index),
                "reward": index,
                "model_id": "behavior",
                "metadata_json": '{"sampling":{"temperature":1}}',
                "datum_json": json.dumps(datum_mapping(datum)),
            }
            for index, datum in enumerate(native)
        ]
        before = restored.forward(native, "cross_entropy").result().loss_fn_outputs
        metrics = ppo_update(
            rows, restored, sdk, behavior_model="behavior", steps=2, learning_rate=0.01
        )
        after = restored.forward(native, "cross_entropy").result().loss_fn_outputs
        assert metrics["updates"] == 2 and metrics["rollouts"] == 2
        assert len(metrics["update_metrics"]) == 2
        for update in metrics["update_metrics"]:
            assert math.isfinite(update["forward_backward"]["loss:sum"])
        assert metrics["update_metrics"][0]["policy_change"]["approx_kl"] == pytest.approx(
            0, abs=1e-8
        )
        preference_before = sum(before[1]["logprobs"].data) - sum(before[0]["logprobs"].data)
        preference_after = sum(after[1]["logprobs"].data) - sum(after[0]["logprobs"].data)
        assert preference_after > preference_before
    finally:
        if provider == "baseten":
            if restored is not None:
                restored.close()
            if original is not None:
                original.close()
        elif provider == "fireworks":
            service.close()
        else:
            service.close("success").result()


@pytest.mark.parametrize("example", ["harbor", "harvey_lab"])
def test_example_train_restores_behavior_and_runs_ppo(example, training_server, tmp_path):
    vocabulary = {
        token: index
        for index, token in enumerate(
            ["[UNK]", "[EOS]", "user", "assistant", ":", "task", "bad", "good"]
        )
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="[EOS]", unk_token="[UNK]"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' }}"
        "{% if message['role'] == 'assistant' %}{% generation %}"
        "{{ message['content'] + eos_token }}{% endgeneration %}"
        "{% else %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    )
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    chats = chat_dataset_from_records(
        [
            {
                "messages": [
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": answer},
                ],
                "group_id": "task",
                "trace_id": str(index),
                "reward": index,
                "model_id": "behavior",
                "metadata_json": '{"sampling":{"temperature":1}}',
            }
            for index, answer in enumerate(["bad", "good"])
        ]
    )
    chats.save_to_disk(str(tmp_path / "chats"))
    baseten = example == "harvey_lab"
    service = (
        loops.ServiceClient(base_url=training_server, api_key="local")
        if baseten
        else tinker.ServiceClient(base_url=training_server, api_key="tml-local")
    )
    client = service.create_lora_training_client(base_model="tiny", rank=2)
    try:
        state = client.save_state("behavior").result()
        receipt = tmp_path / "behavior.json"
        receipt.write_text(
            json.dumps(
                {
                    "state": state.path,
                    "model": "behavior",
                    "base_model": "tiny",
                    "training_url": training_server,
                }
            )
        )
        output = tmp_path / "trained.json"
        script = Path(__file__).resolve().parents[4] / "examples" / example / "run.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "train",
                str(tmp_path / "chats"),
                "--behavior",
                str(receipt),
                "--tokenizer-model",
                str(tmp_path / "tokenizer"),
                "--output",
                str(output),
            ],
            env={**os.environ, "TINKER_API_KEY": "tml-local", "BASETEN_API_KEY": "local"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        trained = json.loads(output.read_text())
        assert trained["rl"]["loss"] == "ppo" and trained["rl"]["rollouts"] == 2
        assert trained["state"] != state.path
    finally:
        if baseten:
            client.close()
        else:
            service.close("success").result()
