import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from trains.datum.adapters.rendering import native_datum


def group_advantages(
    rows: Sequence[Mapping[str, Any]], *, behavior_model: str | None = None
) -> dict[tuple[str, str], float]:
    """Center rewards once per rollout, not once per rendered chunk or model call.

    Examples:
        Multiple calls from one attempt must not change its group baseline:

        >>> rows = [
        ...     {"group_id": "task", "trace_id": "a", "reward": 0.0},
        ...     {"group_id": "task", "trace_id": "a", "reward": 0.0},
        ...     {"group_id": "task", "trace_id": "b", "reward": 1.0},
        ... ]
        >>> group_advantages(rows)
        {('task', 'a'): -0.5, ('task', 'b'): 0.5}
    """
    groups: dict[str, dict[str, float]] = {}
    owners: dict[str, str] = {}
    if not rows:
        raise ValueError("RL requires graded rollout groups")
    for row in rows:
        group, rollout, reward = row.get("group_id"), row.get("trace_id"), row.get("reward")
        if not group or not rollout:
            raise ValueError("RL requires group_id and trace_id for every rollout")
        if (
            not isinstance(reward, (float, int))
            or isinstance(reward, bool)
            or not math.isfinite(reward)
        ):
            raise ValueError(
                "RL requires a finite reward for every rollout; do not filter failures"
            )
        model = row.get("model_id")
        if behavior_model is not None and (
            not isinstance(model, str) or _policy_model(model) != _policy_model(behavior_model)
        ):
            raise ValueError("Rollout model does not match the restored behavior checkpoint")
        if rollout in owners and owners[rollout] != group:
            raise ValueError("One rollout cannot belong to multiple prompt groups")
        owners[rollout] = group
        rewards = groups.setdefault(group, {})
        if rollout in rewards and rewards[rollout] != reward:
            raise ValueError("Conflicting rewards for one rollout")
        rewards[rollout] = float(reward)
    result = {}
    for group, rewards in groups.items():
        if len(rewards) < 2:
            raise ValueError("Each prompt needs at least two independent rollouts")
        mean = math.fsum(rewards.values()) / len(rewards)
        result.update({(group, rollout): reward - mean for rollout, reward in rewards.items()})
    return result


def validate_sampling(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject missing temperatures and sampling truncation incompatible with replay."""
    for row in rows:
        sampling = json.loads(row.get("metadata_json") or "{}").get("sampling") or {}
        if sampling.get("temperature") != 1:
            raise ValueError("Behavior replay requires recorded temperature=1 collection")
        if sampling.get("top_p") not in (None, 1) or sampling.get("top_k") not in (None, 0, -1):
            raise ValueError("Behavior replay requires untruncated sampling (top_p=1, no top_k)")
        if sampling.get("min_p") not in (None, 0):
            raise ValueError("Behavior replay requires min_p=0")


def ppo_update(
    rows: Sequence[Mapping[str, Any]],
    client: Any,
    sdk: Any,
    *,
    behavior_model: str,
    steps: int = 1,
    learning_rate: float = 1e-5,
) -> dict[str, Any]:
    """Apply group-centered PPO to a trainer restored to the collection policy.

    Callers must retain complete model-visible actions and collect from this
    exact checkpoint at temperature=1 without top-p/top-k truncation. Complete
    captured behavior logprobs can be supplied in each datum's logprobs tensor.
    Otherwise frozen teacher-forced references are scored before any update;
    these are distinct from sampling logprobs. No train/serve drift correction
    is claimed for the teacher-forced path.
    Metrics retain each native forward/backward result and optimizer step.
    Policy-change estimates describe the forward pass before its optimizer step.
    """
    if steps < 1 or learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("Positive finite learning rate and update count required")
    advantages = group_advantages(rows, behavior_model=behavior_model)
    validate_sampling(rows)
    if not any(advantages.values()):
        raise ValueError("No within-group reward variation; collect more rollouts, not SFT")
    batch, scale, active_count, reference = _ppo_batch(rows, client, sdk, advantages)
    metrics = {}
    update_metrics = []
    for _ in range(steps):
        result = client.forward_backward(
            batch,
            "ppo",
            loss_fn_config={
                "clip_low_threshold": 0.8,
                "clip_high_threshold": 1.2,
            },
        ).result()
        metrics = (
            client.optim_step(sdk.AdamParams(learning_rate=learning_rate)).result().metrics or {}
        )
        update_metrics.append(
            {
                "forward_backward": result.metrics or {},
                "optimizer": metrics,
                "policy_change": _policy_change(batch, result),
            }
        )
    return {
        "loss": "ppo",
        "advantage": "group_mean_centered",
        "updates": steps,
        "learning_rate": learning_rate,
        "rollouts": len(advantages),
        "groups": len({key[0] for key in advantages}),
        "action_weight": scale,
        "scored_calls": active_count,
        "behavior_reference": reference,
        "optimizer": metrics,
        "update_metrics": update_metrics,
    }


def _ppo_batch(
    rows: Sequence[Mapping[str, Any]],
    client: Any,
    sdk: Any,
    advantages: Mapping[tuple[str, str], float],
) -> tuple[list[Any], float, int, str]:
    """Score complete actions at the frozen behavior checkpoint before any update."""
    data = [native_datum(json.loads(row["datum_json"]), sdk) for row in rows]
    scale = sum(sum(datum.loss_fn_inputs["weights"].data) for datum in data)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("RL batch contains no action tokens")
    # Flat groups have exactly zero PPO gradient. Retain their action count in
    # the global normalization, but avoid scoring and backpropagating their images.
    active = [
        (datum, row)
        for datum, row in zip(data, rows, strict=True)
        if advantages[(row["group_id"], row["trace_id"])] != 0
    ]
    captured = ["logprobs" in datum.loss_fn_inputs for datum in data]
    if any(captured) and not all(captured):
        raise ValueError("Captured behavior logprobs must cover the complete batch")
    if all(captured):
        old = [datum.loss_fn_inputs["logprobs"] for datum, _ in active]
        reference = "sampled"
    else:
        # Forward-only scoring is a teacher-forced reference, not sampling data.
        scored = client.forward([datum for datum, _ in active], "cross_entropy").result()
        old = [output["logprobs"] for output in scored.loss_fn_outputs]
        reference = "teacher_forced"
    batch = [
        _ppo_datum(datum, previous, advantages[(row["group_id"], row["trace_id"])], scale, sdk)
        for (datum, row), previous in zip(active, old, strict=True)
    ]
    return batch, scale, len(active), reference


def _policy_change(batch: Sequence[Any], result: Any) -> dict[str, float] | None:
    """Compare returned action scores with the frozen teacher-forced reference."""
    outputs = getattr(result, "loss_fn_outputs", None)
    if outputs is None or len(outputs) != len(batch):
        return None
    changes = []
    for datum, output in zip(batch, outputs, strict=True):
        values = output.get("logprobs")
        if values is None:
            return None
        old = datum.loss_fn_inputs["logprobs"].data
        advantages = datum.loss_fn_inputs["advantages"].data
        if len(values.data) != len(old):
            return None
        changes.extend(
            new - previous
            for new, previous, advantage in zip(values.data, old, advantages, strict=True)
            if advantage != 0
        )
    if not changes or not all(math.isfinite(value) for value in changes):
        return None
    try:
        ratios = [math.exp(value) for value in changes]
        kl = [math.expm1(value) - value for value in changes]
    except OverflowError:
        return None
    count = len(changes)
    return {
        "action_tokens": float(count),
        "approx_kl": math.fsum(kl) / count,
        "mean_ratio": math.fsum(ratios) / count,
        "ratio_outside_clip_fraction": sum(value < 0.8 or value > 1.2 for value in ratios) / count,
    }


def _policy_model(name: str) -> str:
    """Remove the benchmark provider prefix from a recorded model identity."""
    for prefix in ("openai-api/tinker/", "openai/", "baseten/", "fireworks/"):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def _ppo_datum(datum: Any, old: Any, advantage: float, scale: float, sdk: Any) -> Any:
    """Build an assistant-masked PPO datum from frozen logprobs and an advantage."""
    weights = datum.loss_fn_inputs["weights"].data
    logprobs = list(old.data)
    targets = datum.loss_fn_inputs["target_tokens"]
    if len(logprobs) != len(weights) or len(weights) != len(targets.data):
        raise ValueError(
            "Policy logprobs, masks, and targets must have identical next-token alignment"
        )
    if any(weight and not math.isfinite(lp) for weight, lp in zip(weights, logprobs, strict=True)):
        raise ValueError("Behavior policy logprobs must be finite on action tokens")
    # RL accepts precisely these three tensors. Context, tools and image positions
    # carry zero advantage; a CE weights tensor must not be sent to PPO.
    return sdk.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={
            "target_tokens": targets,
            "logprobs": sdk.TensorData(
                data=[lp if w else 0 for lp, w in zip(logprobs, weights, strict=True)],
                dtype="float32",
                shape=[len(weights)],
            ),
            "advantages": sdk.TensorData(
                data=[advantage * w / scale for w in weights], dtype="float32", shape=[len(weights)]
            ),
        },
    )
