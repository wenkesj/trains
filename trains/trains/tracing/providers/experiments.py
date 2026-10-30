import math
from dataclasses import asdict, dataclass

from trains.formats.tracing import TraceRecord


@dataclass
class ExperimentRun:
    """A native experiment attempt with its task, trace, and selected score."""

    id: str
    trace_id: str
    experiment_id: str
    example_id: str
    dataset_id: str
    dataset_version_id: str | None
    evaluations: list[dict]
    project: str | None = None

    def attach(self, trace: TraceRecord, score_name: str | None) -> TraceRecord:
        """Attach native experiment provenance to a matching trace record."""
        if not self.trace_id or trace.id != self.trace_id or not trace.spans:
            raise ValueError(f"Experiment run {self.id}: missing linked OpenInference trace")
        return trace.model_copy(
            update={
                "metadata": {
                    **trace.metadata,
                    **asdict(self),
                    "run_id": self.id,
                    "reward": _selected_score(self.evaluations, score_name),
                }
            }
        )


def _selected_score(evaluations: list[dict], name: str | None) -> float | None:
    """Select a finite experiment score, requiring a name when ambiguous."""
    if name is None:
        return None
    matches = [e for e in evaluations if e.get("name") == name and not e.get("error")]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one successful evaluation for {name!r}")
    value = matches[0].get("score")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Evaluation {name!r} requires a finite numeric score")
    return float(value)
