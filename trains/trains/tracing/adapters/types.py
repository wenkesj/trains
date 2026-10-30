from typing import TypedDict


class AdapterMatch(TypedDict):
    """The selected adapter and generation count for one source span."""

    span_id: str
    format_id: str
    generations: int
