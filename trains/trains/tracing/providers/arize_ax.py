from collections.abc import Iterator
from typing import Any

from trains.formats.tracing import TraceRecord, TraceSpec
from trains.tracing.providers.client_tracing_provider import ClientTracingProvider
from trains.tracing.providers.experiments import ExperimentRun
from trains.tracing.providers.native import cursor_items, native_mapping
from trains.tracing.providers.utils import traces_from_spans


class ArizeAXTracingProvider(ClientTracingProvider):
    """Read AX spans and native experiments through a caller-owned Arize client."""

    def __init__(
        self,
        client: Any,
        spec: TraceSpec | None = None,
        *,
        experiment: str,
        dataset: str | None = None,
        score_name: str | None = None,
        trace_project: str | None = None,
        space: str | None = None,
    ) -> None:
        """Select a native experiment and the project holding its linked traces.

        Args:
            client: Initialized native provider client.
            spec: Format spec extracting complete model calls.
            experiment: Native experiment identity.
            dataset: Optional dataset identity checked against the experiment.
            score_name: Native evaluator metric supplying terminal rewards.
            trace_project: Native trace project when traces are stored separately.
            space: Native AX space used to resolve named datasets and experiments.
        """
        super().__init__(
            client, spec, experiment=experiment, dataset=dataset, score_name=score_name
        )
        self.trace_project = trace_project
        self.space = space

    def _export_trace(self, run: ExperimentRun) -> Iterator[TraceRecord]:
        """Retrieve only the trace linked to this AX experiment attempt."""
        project = self.trace_project or run.project
        if not project:
            raise ValueError("The AX experiment has no trace project")
        rows = cursor_items(
            self.client.spans.list,
            {
                "project": project,
                **({"space": self.space} if self.space else {}),
                "filter": "context.trace_id = '" + run.trace_id.replace("'", "''") + "'",
            },
            "spans",
        )
        yield from traces_from_spans(rows)

    def iter_experiment_runs(self) -> Iterator[ExperimentRun]:
        """Yield native experiment attempts with their task IDs and rewards."""
        experiment = native_mapping(
            self.client.experiments.get(
                **{
                    "experiment": self.experiment,
                    **({"dataset": self.dataset} if self.dataset else {}),
                    **({"space": self.space} if self.space else {}),
                }
            )
        )
        dataset_id = self._experiment_dataset(experiment)["id"]
        examples = {
            v["id"]
            for v in cursor_items(
                self.client.datasets.list_examples,
                {
                    "dataset": dataset_id,
                    "dataset_version_id": experiment.get("dataset_version_id"),
                },
                "examples",
            )
        }
        for run in cursor_items(
            self.client.experiments.list_runs,
            {"experiment": experiment["id"]},
            "experiment_runs",
        ):
            if run.get("example_id") not in examples:
                continue
            yield ExperimentRun(
                id=run["id"],
                trace_id=run.get("trace_id")
                or run.get("context.trace_id")
                or run.get("result.trace.id")
                or "",
                experiment_id=experiment["id"],
                dataset_id=dataset_id,
                dataset_version_id=experiment.get("dataset_version_id"),
                example_id=run["example_id"],
                evaluations=_run_evaluations(run),
                project=experiment.get("experiment_traces_project_id"),
            )

    def _experiment_dataset(self, experiment: dict) -> dict:
        """Resolve the dataset identity associated with the selected AX experiment."""
        dataset_id = experiment.get("dataset_id")
        if not dataset_id:
            raise ValueError("Select an AX experiment linked to a dataset")
        dataset = native_mapping(
            self.client.datasets.get(
                **{
                    "dataset": self.dataset or dataset_id,
                    **({"space": self.space} if self.space else {}),
                }
            )
        )
        if dataset["id"] != dataset_id:
            raise ValueError("Selected dataset does not belong to the experiment")
        return dataset


def _run_evaluations(run: dict) -> list[dict]:
    """Read native AX evaluation results for an experiment attempt."""
    evaluations = list(run.get("annotations") or [])
    for key, value in run.items():
        if key.startswith("eval.") and key.endswith(".score"):
            evaluations.append({"name": key[5:-6], "score": value})
    return evaluations
