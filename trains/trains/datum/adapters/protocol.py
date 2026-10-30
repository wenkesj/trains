from collections.abc import Iterable
from typing import Any, Protocol


class Adapter[RecordT, DatumT](Protocol):
    """Render full records into provider-native datums, without restricting modalities."""

    def encode(self, record: RecordT, processing_class: Any) -> Iterable[DatumT]:
        """Yield native datums rendered with the supplied tokenizer or processor."""
        ...
