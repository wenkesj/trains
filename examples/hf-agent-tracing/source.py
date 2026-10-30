# Pinned, attributed OpenHands demonstrations for the supervised warm start.

import json

from datasets import load_dataset
from trains.formats.records import validate_record

DATASET = "nvidia/SWE-Hero-openhands-trajectories"
REVISION = "150bc119e52c647216fce285fd801f16b6fd745b"
LICENSE = "CC-BY-4.0"
ATTRIBUTION = "NVIDIA, SWE-Hero Trajectories, version 1.0"
URL = f"https://huggingface.co/datasets/{DATASET}"
REPOSITORY_LICENSES = frozenset({"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"})


def load_conversations(limit):
    """Stream complete trajectories with explicit, retained license provenance."""
    if limit < 1:
        raise ValueError("Select at least one trajectory")
    dataset = load_dataset(DATASET, revision=REVISION, split="train", streaming=True)
    records = [_conversation(row) for row in dataset.take(limit)]
    if not records:
        raise ValueError("No trajectories selected")
    return records


def _conversation(row):
    """Retain attribution and source-repository terms alongside the conversation."""
    if row["license"] not in REPOSITORY_LICENSES:
        raise ValueError(f"Unsupported source repository license: {row['license']!r}")
    messages = _messages(row["trajectory"])
    record = validate_record({"messages": messages})
    metadata = {
        "dataset": DATASET, "revision": REVISION, "dataset_license": LICENSE,
        "attribution": ATTRIBUTION, "url": URL,
        "repository": row["repo"], "repository_license": row["license"],
        "source_dataset": row["dataset"], "instance_id": row["instance_id"],
        "trajectory_id": row["trajectory_id"],
        "tool_result_alignment": "Missing result IDs matched only to a single outstanding call",
    }
    return {
        **record, "trace_id": row["trajectory_id"], "group_id": row["instance_id"],
        "metadata_json": json.dumps(metadata),
    }


def _messages(trajectory):
    """Restore omitted tool result IDs only when the match is unambiguous."""
    messages = []
    pending = set()
    for raw in trajectory:
        message = dict(raw)
        if message["role"] == "tool":
            identifier = message.get("tool_call_id")
            if identifier is None:
                if len(pending) != 1:
                    raise ValueError("Tool result has no unambiguous call ID")
                identifier = next(iter(pending))
                message["tool_call_id"] = identifier
            if identifier not in pending:
                raise ValueError("Orphan or duplicate tool result")
            pending.remove(identifier)
        else:
            if pending:
                raise ValueError("Missing tool results before the next model turn")
            pending = {call["id"] for call in message.get("tool_calls") or []}
        messages.append(message)
    return messages
