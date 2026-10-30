from collections.abc import Iterator
from datetime import datetime
from typing import Any

from trains.formats.tracing import TraceRecord, TraceSpec
from trains.tracing.adapters.span import decoded
from trains.tracing.providers.client_tracing_provider import ClientTracingProvider
from trains.tracing.providers.experiments import ExperimentRun
from trains.tracing.providers.native import cursor_items, native_mapping
from trains.tracing.providers.utils import traces_from_spans


class LangfuseTracingProvider(ClientTracingProvider):
    """Read Langfuse observations and native experiment grades."""

    def __init__(
        self,
        client: Any,
        spec: TraceSpec | None = None,
        *,
        dataset: str,
        experiment: str,
        from_start_time: datetime,
        score_name: str | None = None,
    ) -> None:
        """Select a Langfuse dataset experiment with its required native time bound.

        Args:
            client: Initialized native Langfuse client.
            spec: Format spec extracting complete model calls.
            dataset: Native dataset name.
            experiment: Native experiment ID.
            from_start_time: Time at or before the experiment starts, required by Langfuse.
            score_name: Native evaluator metric supplying terminal rewards.

        Raises:
            ValueError: The dataset is empty or the timestamp has no timezone.
        """
        super().__init__(
            client, spec, dataset=dataset, experiment=experiment, score_name=score_name
        )
        if not dataset or not dataset.strip():
            raise ValueError("Select a Langfuse dataset")
        if from_start_time.tzinfo is None or from_start_time.utcoffset() is None:
            raise ValueError("Langfuse from_start_time requires a timezone")
        self.from_start_time = from_start_time

    def _export_trace(self, run: ExperimentRun) -> Iterator[TraceRecord]:
        """Retrieve only observations linked to this Langfuse experiment attempt."""
        rows = []
        for item in cursor_items(
            self._observation_page,
            {
                "fields": "core,basic,metadata,io",
                "expand_metadata": "attributes",
                "trace_id": run.trace_id,
            },
        ):
            metadata = item.get("metadata") or {}
            attrs = (
                item.get("attributes")
                or metadata.get("attributes")
                or {
                    key.removeprefix("attributes."): value
                    for key, value in metadata.items()
                    if key.startswith("attributes.")
                }
                or metadata
            )
            # Langfuse promotes these original OI values into native I/O fields.
            attrs = decoded(attrs)
            if not isinstance(attrs, dict):
                raise ValueError("Langfuse must retain the original OpenInference attributes")
            attrs = dict(attrs)
            for field in ("input", "output"):
                if field + ".value" not in attrs and item.get(field) is not None:
                    attrs[field + ".value"] = item[field]
            rows.append(
                {
                    "TraceId": item["trace_id"],
                    "SpanId": item["id"],
                    "ParentSpanId": item.get("parent_observation_id"),
                    "SpanName": item.get("name", ""),
                    "status_code": "ERROR" if item.get("level") == "ERROR" else "OK",
                    "SpanAttributes": attrs,
                }
            )
        yield from traces_from_spans(rows)

    def _observation_page(self, **query: Any) -> dict:
        """Fetch one page of Langfuse observations through its native API."""
        page = native_mapping(self.client.api.observations.get_many(**query))
        keys = {
            key
            for item in page["data"]
            for key in (item.get("metadata") or {})
            if key.removeprefix("attributes.").startswith(
                ("openinference.", "llm.", "input.", "output.")
            )
        }
        expanded = set((query.get("expand_metadata") or "").split(","))
        missing = {
            key
            for key in keys
            if not any(
                key == prefix or key.startswith(prefix + ".") for prefix in expanded if prefix
            )
        }
        if missing:
            page = native_mapping(
                self.client.api.observations.get_many(
                    **{
                        **query,
                        "expand_metadata": ",".join(sorted(expanded | missing)),
                    }
                )
            )
        return page

    def iter_experiment_runs(self) -> Iterator[ExperimentRun]:
        """Yield native experiment attempts with their task IDs and rewards."""
        dataset = self._experiment_dataset()
        examples = {item.id for item in dataset.items}
        for run in cursor_items(
            self.client.api.experiments.list_items,
            {
                "from_start_time": self.from_start_time,
                "experiment_id": self.experiment,
                "dataset_id": dataset.id,
                "fields": "core,dataset",
            },
        ):
            example_id = run["experiment_item_id"]
            if example_id not in examples:
                continue
            scores = []
            if self.score_name:
                for score in cursor_items(
                    self.client.api.scores_v3.get_many_v3,
                    {
                        "trace_id": run["trace_id"],
                        "name": self.score_name,
                        "fields": "subject",
                    },
                ):
                    subject = score.get("subject") or {}
                    if (
                        subject.get("kind") == "trace" and subject.get("id") == run["trace_id"]
                    ) or (subject.get("kind") == "observation" and subject.get("id") == run["id"]):
                        scores.append(
                            {"name": score["name"], "score": score["value"], "id": score["id"]}
                        )
            yield ExperimentRun(
                id=run["id"],
                trace_id=run["trace_id"],
                experiment_id=run["experiment_id"],
                example_id=example_id,
                dataset_id=dataset.id,
                dataset_version_id=run.get("experiment_item_version"),
                evaluations=scores,
            )

    def _experiment_dataset(self) -> Any:
        """Resolve the native dataset and require the experiment to belong to it."""
        dataset = self.client.get_dataset(self.dataset)
        experiments = list(
            cursor_items(
                self.client.api.experiments.list,
                {
                    "id": self.experiment,
                    "from_start_time": self.from_start_time,
                    "fields": "core",
                },
            )
        )
        if len(experiments) != 1 or experiments[0]["id"] != self.experiment:
            raise ValueError(
                "Select one existing Langfuse experiment within the supplied time bound"
            )
        if experiments[0].get("dataset_id") != dataset.id:
            raise ValueError("Selected dataset does not belong to the experiment")
        return dataset
