from collections.abc import Iterable, Mapping
from typing import Any

from datasets import Dataset, Features, Json
from trains.datasets.rows import PROVENANCE, provenance
from trains.formats.records import validate_record
from trains.types import ChatRow

FEATURES = Features(
    {
        **PROVENANCE,
        **{key: Json() for key in ("messages", "tools", "chat_template_kwargs")},
    }
)
RECORD_FIELDS = ("messages", "tools", "chat_template_kwargs")


def chat_row(value: Mapping[str, Any]) -> ChatRow:
    """Validate a full conversation and normalize its provenance columns."""
    record = validate_record(
        {key: value[key] for key in RECORD_FIELDS if value.get(key) is not None}
    )
    return {
        "messages": record["messages"],
        "tools": record.get("tools"),
        "chat_template_kwargs": record.get("chat_template_kwargs"),
        **provenance(value),
    }


def chat_dataset_from_records(records: Iterable[Mapping[str, Any]]) -> Dataset:
    """Materialize standard HF/TRL records with optional provenance columns.

    Native HF JSON features preserve multi-turn conversations, tool schemas, and
    structured tool arguments without Arrow inferring a provider-specific schema.

    Args:
        records: Complete HF conversations with optional native provenance columns.

    Returns:
        A reusable Hugging Face dataset preserving structured messages and tools.

    Examples:
        >>> chats = chat_dataset_from_records([{
        ...     "messages": [{"role": "user", "content": "Hi"},
        ...                  {"role": "assistant", "content": "Hello"}],
        ...     "trace_id": "attempt-1", "reward": 0.0,
        ... }])
        >>> chats[0]["trace_id"], chats[0]["reward"]
        ('attempt-1', 0.0)
        >>> chats[0]["messages"][-1]["content"]
        'Hello'
    """
    rows = [chat_row(record) for record in records]
    return Dataset.from_dict(
        {key: [row[key] for row in rows] for key in FEATURES}, features=FEATURES
    )
