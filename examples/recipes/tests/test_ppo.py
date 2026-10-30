import json
import math
from types import SimpleNamespace

import pytest
from recipes.rl import group_advantages, ppo_update
from tinker import types

from trains.datasets.serialization import datum_mapping
from trains.datum.adapters.tinker import TinkerDatumAdapter


def rows():
    datum = TinkerDatumAdapter().from_tokens([1, 2], [3, 4])
    return [
        {
            "group_id": "task",
            "trace_id": trace,
            "reward": reward,
            "model_id": "policy",
            "metadata_json": json.dumps({"sampling": {"temperature": 1}}),
            "datum_json": json.dumps(datum_mapping(datum)),
        }
        for trace, reward in [("bad", 0), ("good", 4)]
    ]


class Client:
    def __init__(self):
        self.forwards = 0
        self.scored = []
        self.batches = []
        self.steps = 0

    def forward(self, data, loss):
        assert loss == "cross_entropy"
        self.forwards += 1
        self.scored.extend(data)
        return SimpleNamespace(
            result=lambda: SimpleNamespace(
                loss_fn_outputs=[
                    {"logprobs": types.TensorData(data=[-9, -1, -2], dtype="float32", shape=[3])}
                    for _ in data
                ]
            )
        )

    def forward_backward(self, data, loss, loss_fn_config):
        assert loss == "ppo"
        assert loss_fn_config == {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
        self.batches.append(data)
        delta = 0 if len(self.batches) == 1 else math.log(1.25)
        return SimpleNamespace(
            result=lambda: SimpleNamespace(
                metrics={"loss:sum": -float(len(self.batches))},
                loss_fn_outputs=[
                    {
                        "logprobs": types.TensorData(
                            data=[-9, -1 + delta, -2 + delta], dtype="float32", shape=[3]
                        )
                    }
                    for _ in data
                ],
            )
        )

    def optim_step(self, params):
        self.steps += 1
        return SimpleNamespace(result=lambda: SimpleNamespace(metrics={"step": self.steps}))


def test_center_once_per_rollout_not_per_model_call_and_keep_failures():
    data = rows()
    result = group_advantages([data[0], data[1], data[1]])
    assert result == {("task", "bad"): -2, ("task", "good"): 2}


def test_ppo_masks_context_drops_ce_weights_and_keeps_old_logprobs_fixed():
    client = Client()
    stats = ppo_update(rows(), client, types, behavior_model="policy", steps=2)
    assert client.forwards == 1 and client.steps == 2
    assert stats["rollouts"] == 2 and stats["loss"] == "ppo"
    bad, good = client.batches[0]
    assert set(bad.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
    assert bad.loss_fn_inputs["advantages"].data == [0, -0.5, -0.5]
    assert good.loss_fn_inputs["advantages"].data == [0, 0.5, 0.5]
    assert bad.loss_fn_inputs["logprobs"].data == [0, -1, -2]
    assert client.batches[1] is client.batches[0]
    assert [
        {key: update[key] for key in ("forward_backward", "optimizer")}
        for update in stats["update_metrics"]
    ] == [
        {"forward_backward": {"loss:sum": -1.0}, "optimizer": {"step": 1}},
        {"forward_backward": {"loss:sum": -2.0}, "optimizer": {"step": 2}},
    ]
    first, second = [update["policy_change"] for update in stats["update_metrics"]]
    assert first == {
        "action_tokens": 4.0,
        "approx_kl": 0.0,
        "mean_ratio": 1.0,
        "ratio_outside_clip_fraction": 0.0,
    }
    assert second["action_tokens"] == 4
    assert second["mean_ratio"] == pytest.approx(1.25)
    assert second["approx_kl"] == pytest.approx(0.25 - math.log(1.25))
    assert second["ratio_outside_clip_fraction"] == 1


@pytest.mark.parametrize(
    "change,match",
    [
        ({"reward": None}, "finite reward"),
        ({"reward": float("nan")}, "finite reward"),
        ({"model_id": "another-policy"}, "behavior checkpoint"),
        ({"group_id": None}, "group_id"),
        ({"metadata_json": '{"sampling":{"temperature":0.5}}'}, "temperature=1"),
        ({"metadata_json": '{"sampling":{"temperature":1,"top_p":0.9}}'}, "untruncated"),
    ],
)
def test_invalid_or_off_policy_input_stops_before_updates(change, match):
    data = rows()
    data[0].update(change)
    client = Client()
    with pytest.raises(ValueError, match=match):
        ppo_update(data, client, types, behavior_model="policy")
    assert client.steps == 0 and client.forwards == 0


def test_singletons_conflicting_rewards_and_flat_groups_do_not_fall_back_to_sft():
    data = rows()
    with pytest.raises(ValueError, match="two independent"):
        group_advantages(data[:1])
    with pytest.raises(ValueError, match="Conflicting"):
        group_advantages([*data, {**data[1], "reward": 7}])
    client = Client()
    data[0]["reward"] = 4
    with pytest.raises(ValueError, match="No within-group"):
        ppo_update(data, client, types, behavior_model="policy")
    assert client.steps == 0


def test_flat_groups_keep_global_normalization_without_model_work():
    data = rows()
    flat = [
        dict(row, group_id="flat", trace_id="flat-" + row["trace_id"], reward=1) for row in data
    ]
    client = Client()
    stats = ppo_update([*data, *flat], client, types, behavior_model="policy")
    assert stats["rollouts"] == 4 and stats["groups"] == 2
    assert stats["action_weight"] == 8
    assert stats["scored_calls"] == len(client.scored) == 2
    bad, good = client.batches[0]
    assert bad.loss_fn_inputs["advantages"].data == [0, -0.25, -0.25]
    assert good.loss_fn_inputs["advantages"].data == [0, 0.25, 0.25]


def test_actual_captured_references_are_used_without_rescoring():
    data = rows()
    for row in data:
        value = json.loads(row["datum_json"])
        value["loss_fn_inputs"]["logprobs"] = {
            "data": [0, -3, -4],
            "dtype": "float32",
            "shape": [3],
        }
        row["datum_json"] = json.dumps(value)
    client = Client()
    result = ppo_update(data, client, types, behavior_model="policy", steps=2)
    assert result["behavior_reference"] == "sampled"
    assert client.forwards == 0 and client.steps == 2
    assert client.batches[0][0].loss_fn_inputs["logprobs"].data == [0, -3, -4]
    assert client.batches[1] is client.batches[0]
    value = json.loads(data[1]["datum_json"])
    value["loss_fn_inputs"].pop("logprobs")
    data[1]["datum_json"] = json.dumps(value)
    client = Client()
    with pytest.raises(ValueError, match="complete batch"):
        ppo_update(data, client, types, behavior_model="policy")
    assert client.forwards == 0 and client.steps == 0
