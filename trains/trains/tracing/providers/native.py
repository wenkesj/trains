from collections.abc import Callable, Iterator
from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic_core import to_jsonable_python


def native_mapping(value: Any) -> dict:
    """Retain generated SDK read-only fields and dynamic additional properties."""
    if isinstance(value, dict):
        extra = value.get("additional_properties") or {}
        return {
            **{key: item for key, item in value.items() if key != "additional_properties"},
            **extra,
        }
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "model_dump"):
        # Some generated SDKs force wire aliases in their JSON serializer even
        # with by_alias=False. Read native field names before JSON conversion.
        result = to_jsonable_python(value.model_dump(mode="python", by_alias=False))
        extra = result.pop("additional_properties", {})
        return {**result, **extra}
    return value.to_dict()


def cursor_items(fetch: Callable[..., Any], query: dict, key: str = "data") -> Iterator[dict]:
    """Read a complete native cursor collection and reject truncated/repeating pages."""
    if query.get("cursor"):
        raise ValueError("Start a collection at its first page, without a cursor")
    cursor, visited = None, set()
    while True:
        page = native_mapping(fetch(**{**query, **({"cursor": cursor} if cursor else {})}))
        for value in page[key] or []:
            yield native_mapping(value)
        pagination = page.get("pagination") or page.get("meta") or {}
        cursor = pagination.get("next_cursor") or pagination.get("cursor")
        if not cursor:
            if pagination.get("has_more"):
                raise ValueError("Provider reported more results without a continuation cursor")
            return
        if cursor in visited:
            raise ValueError("Provider repeated a pagination cursor")
        visited.add(cursor)
