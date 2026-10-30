from typing import Protocol

from trains.formats.records import HFRecord
from trains.tracing.adapters.span import Span


class TraceAdapter(Protocol):
    """Translate one provider-native span into standard HF prompt/completion records."""

    @property
    def id(self) -> str:
        """Return the stable format identifier used by the adapter registry."""
        ...

    @property
    def description(self) -> str:
        """Describe the native payload format accepted by this adapter."""
        ...

    def matches(self, span: Span) -> bool:
        """Return whether the span contains the model-call format handled here."""
        ...

    def extract(self, span: Span) -> list[HFRecord]:
        """Convert the matched span into complete HF model-call records."""
        ...
