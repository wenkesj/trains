from collections.abc import Iterator
from typing import Any

from trains.formats.tracing import Identity, Reward, TraceRecord, TraceSpec
from trains.tracing.providers.experiments import ExperimentRun


class ClientTracingProvider:
    """Export model calls linked to runs of one native dataset experiment."""

    def __init__(
        self,
        client: Any,
        spec: TraceSpec | None = None,
        *,
        experiment: str,
        dataset: str | None = None,
        score_name: str | None = None,
    ) -> None:
        """Bind a native experiment and optional dataset identity.

        Args:
            client: Initialized native provider client owned by the application.
            spec: Format spec extracting complete model calls.
            experiment: Native experiment identity; trace-only exports are unsupported.
            dataset: Optional dataset identity to validate against the experiment.
            score_name: Native evaluator metric supplying the terminal reward.

        Raises:
            ValueError: The experiment identity is empty.
        """
        if not experiment or not experiment.strip():
            raise ValueError("Select a native experiment to retrieve actual dataset executions")
        self.client, self.spec = client, spec or TraceSpec()
        self.dataset, self.experiment, self.score_name = dataset, experiment, score_name
        self.reward: Reward | None = None
        self.group_id: Identity | None = None
        self.model_id: Identity | None = None

    def iter_traces(self) -> Iterator[TraceRecord]:
        """Yield only complete traces linked to the selected experiment's runs."""
        seen = set()
        for run in self.iter_experiment_runs():
            if not run.trace_id:
                raise ValueError(f"Experiment run {run.id}: missing linked OpenInference trace")
            if run.trace_id in seen:
                raise ValueError(
                    "Multiple experiment executions share a trace; selection is ambiguous"
                )
            seen.add(run.trace_id)
            traces = list(self._export_trace(run))
            if len(traces) != 1:
                raise ValueError(f"Experiment run {run.id}: expected one complete linked trace")
            yield run.attach(traces[0], self.score_name)

    def iter_experiment_runs(self) -> Iterator[ExperimentRun]:
        """Yield native experiment attempts with their task IDs and rewards."""
        raise NotImplementedError

    def _export_trace(self, run: ExperimentRun) -> Iterator[TraceRecord]:
        """Retrieve the complete trace linked to one native experiment run."""
        raise NotImplementedError
