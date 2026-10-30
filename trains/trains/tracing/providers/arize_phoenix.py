from collections.abc import AsyncIterator, Iterator
from typing import Any

from trains.formats.tracing import TraceRecord, TraceSpec
from trains.tracing.providers.client_tracing_provider import ClientTracingProvider
from trains.tracing.providers.experiments import ExperimentRun
from trains.tracing.providers.native import native_mapping
from trains.tracing.providers.utils import frame_records, traces_from_spans

try:
    from phoenix.client.types.spans import SpanQuery as _SpanQuery
except ImportError as error:
    _SpanQuery = None
    _PHOENIX_IMPORT_ERROR = error
else:
    _PHOENIX_IMPORT_ERROR = None


class ArizePhoenixTracingProvider(ClientTracingProvider):
    """Read Phoenix spans and experiment grades through its native client."""

    def __init__(
        self,
        client: Any,
        spec: TraceSpec | None = None,
        *,
        experiment: str,
        dataset: str | None = None,
        score_name: str | None = None,
        trace_project: str | None = None,
    ) -> None:
        """Select a native experiment and the project holding its linked traces.

        Args:
            client: Initialized native provider client.
            spec: Format spec extracting complete model calls.
            experiment: Native experiment identity.
            dataset: Optional dataset identity checked against the experiment.
            score_name: Native evaluator metric supplying terminal rewards.
            trace_project: Native trace project when traces are stored separately.
        """
        if _SpanQuery is None:
            raise ImportError(
                "Phoenix tracing requires arize-phoenix-client. Install it with pip install 'trains-sdk[arize-phoenix]'."
            ) from _PHOENIX_IMPORT_ERROR
        super().__init__(
            client, spec, experiment=experiment, dataset=dataset, score_name=score_name
        )
        self.trace_project = trace_project

    def _export_trace(self, run: ExperimentRun) -> Iterator[TraceRecord]:
        """Retrieve only the trace linked to this Phoenix experiment attempt."""
        frame = self.client.spans.get_spans_dataframe(**self._trace_arguments(run))
        if frame is not None:
            yield from traces_from_spans(frame_records(frame))

    def _trace_arguments(self, run: ExperimentRun) -> dict:
        """Build the native trace-ID lookup without exposing arbitrary filters."""
        if _SpanQuery is None:
            raise ImportError(
                "Phoenix tracing requires arize-phoenix-client. Install it with pip install 'trains-sdk[arize-phoenix]'."
            ) from _PHOENIX_IMPORT_ERROR
        project = self.trace_project or run.project
        if not project:
            raise ValueError("The Phoenix experiment has no trace project")
        return {
            "project_identifier": project,
            "query": _SpanQuery().where(f"context.trace_id == {run.trace_id!r}"),
        }

    def iter_experiment_runs(self) -> Iterator[ExperimentRun]:
        """Yield native experiment attempts with their task IDs and rewards."""
        experiment = native_mapping(
            self.client.experiments.get_experiment(experiment_id=self.experiment)
        )
        dataset = self.client.datasets.get_dataset(
            dataset=self.dataset or experiment["dataset_id"],
            version_id=experiment["dataset_version_id"],
        )
        yield from self._runs_from_experiment(experiment, dataset)

    def _runs_from_experiment(self, experiment: dict, dataset: Any) -> Iterator[ExperimentRun]:
        """Normalize Phoenix experiment attempts with their dataset identities."""
        if dataset.id != experiment["dataset_id"]:
            raise ValueError("Selected dataset does not belong to the experiment")
        examples = {}
        for example in dataset.examples:
            value = native_mapping(example)
            examples[value["id"]] = value
            if value.get("node_id"):
                examples[value["node_id"]] = value
        evaluations = [native_mapping(value) for value in experiment["evaluation_runs"]]
        for value in experiment["task_runs"]:
            run = native_mapping(value)
            if run["dataset_example_id"] not in examples:
                continue
            scores = []
            for evaluation in evaluations:
                if evaluation["experiment_run_id"] == run["id"]:
                    result = evaluation.get("result") or {}
                    if isinstance(result, list):
                        scores.extend(
                            {"name": evaluation["name"], **v, "error": evaluation.get("error")}
                            for v in result
                        )
                    else:
                        scores.append(
                            {"name": evaluation["name"], **result, "error": evaluation.get("error")}
                        )
            yield ExperimentRun(
                id=run["id"],
                trace_id=run.get("trace_id") or "",
                experiment_id=experiment["experiment_id"],
                dataset_id=experiment["dataset_id"],
                dataset_version_id=experiment["dataset_version_id"],
                example_id=examples[run["dataset_example_id"]]["id"],
                evaluations=scores,
                project=experiment.get("project_name"),
            )

    async def aiter_traces(self) -> AsyncIterator[TraceRecord]:
        """Asynchronously yield native traces with selected provenance."""
        experiment = native_mapping(
            await self.client.experiments.get_experiment(
                experiment_id=self.experiment,
            )
        )
        dataset = await self.client.datasets.get_dataset(
            dataset=self.dataset or experiment["dataset_id"],
            version_id=experiment["dataset_version_id"],
        )
        seen = set()
        for run in self._runs_from_experiment(experiment, dataset):
            if not run.trace_id:
                raise ValueError(f"Experiment run {run.id}: missing linked OpenInference trace")
            if run.trace_id in seen:
                raise ValueError(
                    "Multiple experiment executions share a trace; selection is ambiguous"
                )
            seen.add(run.trace_id)
            frame = await self.client.spans.get_spans_dataframe(**self._trace_arguments(run))
            traces = list(traces_from_spans(frame_records(frame))) if frame is not None else []
            if len(traces) != 1:
                raise ValueError(f"Experiment run {run.id}: expected one complete linked trace")
            yield run.attach(traces[0], self.score_name)
