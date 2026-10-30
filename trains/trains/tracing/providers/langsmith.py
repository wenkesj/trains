import json
from collections.abc import Iterator
from itertools import batched
from typing import Any

from trains.formats.tracing import TraceRecord, TraceSpec
from trains.tracing.providers.client_tracing_provider import ClientTracingProvider
from trains.tracing.providers.experiments import ExperimentRun
from trains.tracing.providers.native import native_mapping
from trains.tracing.providers.utils import traces_from_spans


class LangSmithTracingProvider(ClientTracingProvider):
    """Read LangSmith model runs and native experiment feedback."""

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
        super().__init__(
            client, spec, experiment=experiment, dataset=dataset, score_name=score_name
        )
        self.trace_project = trace_project

    def _export_trace(self, run: ExperimentRun) -> Iterator[TraceRecord]:
        """Retrieve only the trace linked to this LangSmith experiment attempt."""
        rows = []
        for value_run in self.client.list_runs(
            trace_id=run.trace_id,
            **({"project_name": self.trace_project} if self.trace_project else {}),
        ):
            rows.append(_run_span(native_mapping(value_run)))
        yield from traces_from_spans(rows)

    def iter_experiment_runs(self) -> Iterator[ExperimentRun]:
        """Yield native experiment attempts with their task IDs and rewards."""
        project = native_mapping(self.client.read_project(project_name=self.experiment))
        dataset_id = str(project.get("reference_dataset_id") or "")
        if not dataset_id:
            raise ValueError("Select a LangSmith experiment linked to a dataset")
        if self.dataset:
            dataset = native_mapping(self.client.read_dataset(dataset_name=self.dataset))
            if str(dataset["id"]) != dataset_id:
                raise ValueError("Selected dataset does not belong to the experiment")
        version = (project.get("extra") or {}).get("metadata", {}).get("dataset_version")
        examples = {
            str(native_mapping(v)["id"])
            for v in self.client.list_examples(
                **{
                    "as_of": version,
                    "dataset_id": dataset_id,
                }
            )
        }
        for batch in batched(self.client.list_runs(project_id=project["id"], is_root=True), 64):
            runs = [native_mapping(value) for value in batch]
            runs = [run for run in runs if str(run.get("reference_example_id") or "") in examples]
            if not runs:
                continue
            feedback = self._run_feedback([str(run["id"]) for run in runs])
            trace_ids = self._trace_ids(runs)
            for run in runs:
                yield ExperimentRun(
                    id=str(run["id"]),
                    trace_id=trace_ids[str(run["id"])],
                    experiment_id=str(project["id"]),
                    example_id=str(run["reference_example_id"]),
                    dataset_id=dataset_id,
                    dataset_version_id=version,
                    evaluations=feedback[str(run["id"])],
                )

    def _trace_ids(self, runs: list[dict[str, Any]]) -> dict[str, str]:
        """Resolve a bounded batch of experiment links using only native trace metadata."""
        resolved: dict[str, str] = {}
        pending: dict[str, list[str]] = {}
        for run in runs:
            metadata = (run.get("extra") or {}).get("metadata") or {}
            otel_id = metadata.get("OTEL_TRACE_ID")
            if self.trace_project and otel_id:
                pending.setdefault(str(otel_id), []).append(str(run["id"]))
            else:
                resolved[str(run["id"])] = str(run["trace_id"])
        if pending:
            values = list(pending)
            selector = (
                "eq(metadata_value," + json.dumps(values[0]) + ")"
                if len(values) == 1
                else "in(metadata_value," + json.dumps(values) + ")"
            )
            query = 'and(eq(metadata_key,"OTEL_TRACE_ID"),' + selector + ")"
            found: dict[str, set[str]] = {otel_id: set() for otel_id in pending}
            for value in self.client.list_runs(
                project_name=self.trace_project,
                filter=query,
                select=["id", "trace_id", "extra"],
            ):
                linked = native_mapping(value)
                metadata = (linked.get("extra") or {}).get("metadata") or {}
                otel_id = str(metadata.get("OTEL_TRACE_ID") or "")
                if otel_id in found and linked.get("trace_id"):
                    found[otel_id].add(str(linked["trace_id"]))
            for otel_id, run_ids in pending.items():
                if len(found[otel_id]) != 1:
                    raise ValueError(
                        "The experiment's OpenTelemetry trace is missing or ambiguous in LangSmith"
                    )
                trace_id = next(iter(found[otel_id]))
                resolved.update((run_id, trace_id) for run_id in run_ids)
        return resolved

    def _run_feedback(self, run_ids: list[str]) -> dict[str, list[dict]]:
        """Fetch a bounded batch of native feedback, preserving its run identities."""
        scores: dict[str, list[dict]] = {run_id: [] for run_id in run_ids}
        if self.score_name:
            for value in self.client.list_feedback(run_ids=run_ids, feedback_key=[self.score_name]):
                feedback = native_mapping(value)
                run_id = str(feedback.get("run_id"))
                if run_id in scores:
                    scores[run_id].append(
                        {
                            "name": feedback["key"],
                            "score": feedback.get("score"),
                            "id": str(feedback["id"]),
                        }
                    )
        return scores


def _run_span(value: dict[str, Any]) -> dict[str, Any]:
    """Restore OpenInference fields that LangSmith stores as native run data."""
    extra = value.get("extra") or {}
    metadata = extra.get("metadata") or {}
    attributes = dict(value.get("attributes") or metadata.get("attributes") or metadata)
    attributes.update(
        {
            key.removeprefix("otel.span."): item
            for key, item in metadata.items()
            if key.startswith("otel.span.")
        }
    )
    attributes.setdefault("openinference.span.kind", str(value.get("run_type", "chain")).upper())
    for source, target in (("inputs", "input.value"), ("outputs", "output.value")):
        if value.get(source) is not None:
            attributes.setdefault(target, value[source])
    if extra.get("invocation_params") is not None:
        attributes.setdefault("llm.invocation_parameters", extra["invocation_params"])
    if metadata.get("ls_model_name") is not None:
        attributes.setdefault("llm.model_name", metadata["ls_model_name"])
    if metadata.get("ls_provider") is not None:
        attributes.setdefault("llm.provider", metadata["ls_provider"])
    return {
        "TraceId": str(value["trace_id"]),
        "SpanId": str(value["id"]),
        "ParentSpanId": str(value.get("parent_run_id") or ""),
        "SpanName": value.get("name", ""),
        "SpanAttributes": attributes,
        "status_code": "ERROR" if value.get("error") else "OK",
    }
