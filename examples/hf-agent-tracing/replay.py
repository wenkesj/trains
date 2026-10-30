# Validate native sessions against token receipts and build the Loops RL batch.

import json

from datasets import concatenate_datasets
from recipes.rl import group_advantages, ppo_update
from trains.datasets import chat_dataset_from_sessions, datum_dataset_from_datums
from trains.datum.adapters.baseten import BasetenDatumAdapter
from trains.sessions.adapters import PiSessionAdapter


def canonical_output(message):
    """Compare persisted Pi actions without provider-only message metadata."""
    result = {
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "tool_calls": [],
    }
    for call in message.get("tool_calls") or []:
        function = dict(call["function"])
        if isinstance(function["arguments"], str):
            function["arguments"] = json.loads(function["arguments"])
        result["tool_calls"].append({"id": call["id"], "function": function})
    return result


def session_datums(collection, policy):
    """Import every rollout and retain its exact generated action tokens.

    Pi sessions are the conversation source. Model-boundary receipts supplement
    them with tokenizer-exact inputs/actions and the system prompt/tool schemas
    that Pi does not persist in its session messages. A missing or changed action
    rejects the entire batch instead of silently filtering a failed rollout.
    """
    manifest = json.loads((collection / "collection.json").read_text())
    if manifest["model"] != policy["model"]:
        raise ValueError("Collection was sampled from a different checkpoint")
    chats, datums = [], []
    receipts = [
        json.loads((collection / attempt / "receipt.json").read_text())
        for attempt in manifest["attempts"]
    ]
    captured = [
        call.get("sampling_logprobs") is not None
        for receipt in receipts
        for call in receipt["calls"]
    ]
    if any(captured) and not all(captured):
        raise ValueError("Captured sampling logprobs must cover the complete collection")
    adapter = BasetenDatumAdapter()
    for attempt in manifest["attempts"]:
        receipt = json.loads((collection / attempt / "receipt.json").read_text())
        if receipt["model"] != policy["model"] or not receipt["calls"]:
            raise ValueError(f"Attempt {attempt} has no usable policy actions")
        chat = chat_dataset_from_sessions(receipt["session_dir"], adapter=PiSessionAdapter())
        if len(chat) != 1:
            raise ValueError("Expected exactly one Pi session per attempt")
        actions = [message for message in chat[0]["messages"] if message["role"] == "assistant"]
        calls = receipt["calls"]
        if [canonical_output(m) for m in actions] != [canonical_output(c["output"]) for c in calls]:
            raise ValueError(f"Attempt {attempt}: session actions differ from sampled actions")
        if any(call["checkpoint"] != policy["model"] for call in calls):
            raise ValueError("A sampled call used a different policy")
        provenance = _provenance(attempt, receipt, policy)
        chats.append(chat.map(lambda row, provenance=provenance: provenance))
        datums.append(_native_datums(calls, adapter, provenance))
    if not datums:
        raise ValueError("Collection contains no attempts")
    return concatenate_datasets(chats), concatenate_datasets(datums)


def reinforce(collection, policy, service, sdk, *, steps=1, learning_rate=1e-5):
    """Start PPO with fresh Adam state and resume it across later PPO rounds."""
    chats, datums = session_datums(collection, policy)
    # Validate grades before allocating a trainer; every failure remains in its group.
    advantages = group_advantages(list(chats), behavior_model=policy["model"])
    if not any(advantages.values()):
        raise ValueError("No reward variation: collect a new complete batch with more attempts")
    chats.save_to_disk(str(collection / "chats"))
    datums.save_to_disk(str(collection / "datums"))
    resume_optimizer = "rl_metrics" in policy
    restore = (
        service.create_training_client_from_state_with_optimizer
        if resume_optimizer
        else service.create_training_client_from_state
    )
    client = restore(policy["state"])
    try:
        metrics = ppo_update(
            list(datums),
            client,
            sdk,
            behavior_model=policy["model"],
            steps=steps,
            learning_rate=learning_rate,
        )
        metrics["optimizer_state"] = "resumed_ppo" if resume_optimizer else "fresh_ppo"
        name = "hf-trace-rl-" + collection.name
        state = client.save_state(name).result()
        sampler = client.save_weights_for_sampler(name).result()
        receipt = {
            **policy,
            "state": state.path,
            "model": sampler.path,
            "previous_model": policy["model"],
            "rl_metrics": metrics,
        }
        (collection / "policy.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    finally:
        client.close()


def _provenance(attempt, receipt, policy):
    """Provenance."""
    provenance = {
        "trace_id": attempt,
        "group_id": receipt["group_id"],
        "model_id": policy["model"],
        "reward": receipt["reward"],
        "metadata_json": json.dumps(
            {
                "sampling": receipt["sampling"],
                "session_dir": receipt["session_dir"],
                "exit_code": receipt["exit_code"],
            }
        ),
    }
    return provenance


def _native_datums(calls, adapter, provenance):
    """Retain aligned native action tokens and captured behavior probabilities."""
    native = []
    for call in calls:
        datum = adapter.from_tokens(call["prompt_tokens"], call["completion_tokens"])
        if call.get("sampling_logprobs") is not None:
            probabilities = call["sampling_logprobs"]
            if len(probabilities) != len(call["completion_tokens"]):
                raise ValueError("Sampling logprobs and captured action tokens differ in length")
            datum.loss_fn_inputs["logprobs"] = type(datum.loss_fn_inputs["weights"])(
                data=[0.0] * (len(call["prompt_tokens"]) - 1) + probabilities,
                dtype="float32",
                shape=[len(datum.loss_fn_inputs["weights"].data)],
            )
        native.append(datum)
    return datum_dataset_from_datums(native).map(lambda row, provenance=provenance: provenance)
