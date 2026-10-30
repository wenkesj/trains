"""Exercise the native Pi CLI and replay safety without remote GPU allocation."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from baseten import loops
from recipes.exercism import EXERCISM_ROOT, Exercise
from recipes.models import DEFAULT_MODEL, load_tokenizer

from bridge import wire_message
from collection import _grade, collect_one
from rendering import LiquidRenderer
from replay import reinforce, session_datums

IMAGE = "trains-pi-harness:0.73.1"


class ScriptedSampler:
    def __init__(self, tokenizer, correct, *, filename="solution.py", solution=None):
        code = (
            "def count_vowels(text):\n    return sum(c in 'aeiou' for c in text.lower())\n"
            if correct
            else "def count_vowels(text):\n    return 0\n"
        )
        if solution is not None:
            code = solution
        self.tokenizer = tokenizer
        self.responses = [
            f"</think><tool_call>\n<function=read>\n<parameter=path>{filename}</parameter>\n</function>\n</tool_call>",
            f"</think><tool_call>\n<function=write>\n<parameter=path>{filename}</parameter>\n<parameter=content>"
            + code
            + "</parameter>\n</function>\n</tool_call>",
            "</think>Done.",
        ]

    def sample(self, prompt, sampling_params):
        assert sampling_params.temperature == 1 and sampling_params.top_p == 1
        assert sampling_params.top_k == -1
        tokens = self.tokenizer.encode(
            self.responses.pop(0) + "<|im_end|>", add_special_tokens=False
        )
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=tokens, stop_reason="stop")])


class Trainer:
    """Capture native Loops tensors and updates while supplying deterministic scores."""

    def __init__(self):
        self.updates = 0
        self.closed = False

    def forward(self, batch, loss):
        assert loss == "cross_entropy"
        outputs = [
            {
                "logprobs": loops.TensorData(
                    data=[-1.0] * len(d.loss_fn_inputs["weights"].data),
                    dtype="float32",
                    shape=[len(d.loss_fn_inputs["weights"].data)],
                )
            }
            for d in batch
        ]
        return SimpleNamespace(result=lambda: SimpleNamespace(loss_fn_outputs=outputs))

    def forward_backward(self, batch, loss, loss_fn_config):
        assert loss == "ppo"
        self.batch = batch
        return SimpleNamespace(result=lambda: SimpleNamespace(metrics={"ppo": 1}))

    def optim_step(self, params):
        self.updates += 1
        return SimpleNamespace(result=lambda: SimpleNamespace(metrics={"step": self.updates}))

    def save_state(self, name):
        return SimpleNamespace(result=lambda: SimpleNamespace(path="bt://rl-state"))

    def save_weights_for_sampler(self, name):
        return SimpleNamespace(result=lambda: SimpleNamespace(path="bt://rl-sampler"))

    def close(self):
        self.closed = True


@pytest.mark.parametrize("resume", [False, True])
def test_native_sessions_to_baseten_ppo_and_reject_changed_actions(tmp_path, resume):
    available = subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True, check=False
    )
    if available.returncode:
        pytest.skip("Build the native Pi harness image before running this integration test")
    tokenizer = load_tokenizer(DEFAULT_MODEL)
    renderer = LiquidRenderer(tokenizer)
    policy = {"model": "bt://sft-sampler", "state": "bt://sft-state"}
    if resume:
        policy["rl_metrics"] = {"updates": 1}
    receipts = []
    for attempt, correct in [("bad", False), ("good", True)]:
        receipts.append(
            collect_one(
                tmp_path / attempt,
                tmp_path / "sessions",
                policy,
                ScriptedSampler(tokenizer, correct),
                renderer,
                image=IMAGE,
            )
        )
    assert [r["reward"] for r in receipts] == [0.0, 1.0]
    assert all(r["exit_code"] == 0 and len(r["calls"]) == 3 and not r["errors"] for r in receipts)
    (tmp_path / "collection.json").write_text(
        json.dumps({"model": policy["model"], "attempts": ["bad", "good"]})
    )
    chats, datums = session_datums(tmp_path, policy)
    assert len(chats) == 2 and len(datums) == 6
    trainer = Trainer()

    def restore(state):
        assert state == policy["state"]
        return trainer

    method = (
        "create_training_client_from_state_with_optimizer"
        if resume
        else "create_training_client_from_state"
    )
    service = SimpleNamespace(**{method: restore})
    result = reinforce(tmp_path, policy, service, loops)
    assert result["model"] == "bt://rl-sampler"
    assert trainer.updates == 1 and trainer.closed
    assert result["rl_metrics"]["optimizer_state"] == ("resumed_ppo" if resume else "fresh_ppo")
    advantages = [sum(d.loss_fn_inputs["advantages"].data) for d in trainer.batch]
    assert min(advantages) < 0 < max(advantages)
    # Changing a persisted action must fail before another optimizer update.
    session = next(Path(receipts[0]["session_dir"]).glob("*.jsonl"))
    session.write_text(session.read_text().replace("Done.", "Changed."))
    with pytest.raises(ValueError, match="session actions differ"):
        reinforce(tmp_path, policy, service, loops)
    assert trainer.updates == 1


def test_grader_cannot_read_host_files_or_credentials_and_has_no_network(tmp_path, monkeypatch):
    if subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True, check=False
    ).returncode:
        pytest.skip("Build the native Pi harness image first")
    monkeypatch.setenv("TRAINS_HARNESS_SENTINEL", "must-stay-on-host")
    host_secret = tmp_path / "host-secret"
    host_secret.write_text("must-stay-on-host")
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "solution.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "assert os.getenv('TRAINS_HARNESS_SENTINEL') is None\n"
        f"assert not Path({str(host_secret)!r}).exists()\n"
        "assert sorted(os.listdir('/sys/class/net')) == ['lo']\n"
        "def count_vowels(text):\n    return sum(c in 'aeiou' for c in text.lower())\n"
    )
    reward, error = _grade(workspace, image=IMAGE)
    assert reward == 1, error


def test_native_pi_uses_upstream_exercism_tests_only_in_the_verifier(tmp_path):
    if subprocess.run(
        ["docker", "image", "inspect", IMAGE], capture_output=True, check=False
    ).returncode:
        pytest.skip("Build the native Pi harness image first")
    exercise = Exercise.load(EXERCISM_ROOT, "two-fer")
    tokenizer = load_tokenizer(DEFAULT_MODEL)
    receipt = collect_one(
        tmp_path / "native-exercise",
        tmp_path / "sessions",
        {"model": "bt://frozen", "base_model": DEFAULT_MODEL},
        ScriptedSampler(
            tokenizer,
            True,
            filename="two_fer.py",
            solution="def two_fer(name='you'):\n    return f'One for {name}, one for me.'\n",
        ),
        LiquidRenderer(tokenizer),
        image=IMAGE,
        exercise=exercise,
    )
    assert receipt["reward"] == 1
    assert receipt["group_id"] == exercise.slug
    assert receipt["benchmark"] == exercise.manifest()
    assert json.loads(receipt["grading"])["tests"] == 3
    assert not (tmp_path / "native-exercise/workspace/two_fer_test.py").exists()
    assert (tmp_path / "native-exercise/verifier/two_fer_test.py").is_file()


def test_liquid_thinking_prefill_is_never_invented_in_the_model_response():
    tokenizer = load_tokenizer(DEFAULT_MODEL)
    renderer = LiquidRenderer(tokenizer)
    empty, complete = renderer.parse_response(
        tokenizer.encode("</think>Done.<|im_end|>", add_special_tokens=False)
    )
    wire = wire_message(empty)
    assert complete.is_clean and wire["content"] == "Done."
    assert not wire.get("reasoning_content")
    reasoned, _ = renderer.parse_response(
        tokenizer.encode("Consider it.</think>Done.<|im_end|>", add_special_tokens=False)
    )
    assert wire_message(reasoned)["reasoning_content"] == "Consider it."
    unfinished, incomplete = renderer.parse_response(
        tokenizer.encode("Still thinking", add_special_tokens=False)
    )
    assert not incomplete.is_clean
    assert wire_message(unfinished)["content"] == "Still thinking"
