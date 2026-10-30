# Use AX's native dataset/experiment upload and span export APIs.

import json
import time
from datetime import UTC, datetime
from uuid import uuid4

from arize.experiments.types import ExperimentTaskFieldNames
from recipes.tracing import has_completed_output
from trains.datasets import chat_dataset_from_tracing_provider
from trains.tracing import ArizeAXTracingProvider
from trains.tracing.providers.native import cursor_items
from trains.tracing.providers.utils import traces_from_spans

from openai_tracing.spec import MODEL_CALL_SPANS, ModelCallSpec
from openai_tracing.workflow import attempt


class AXBackend:
    """Collect and export graded coding experiments through the Arize AX client."""

    def __init__(self, client, *, space: str, project: str) -> None:
        """Bind native AX collection to its space and trace project.

        Args:
            client: Initialized native provider client owned by the application.
            space: Native AX space identity.
            project: Native trace destination project.
        """
        self.client, self.space, self.project = client, space, project

    def collect(self, exercises, policy, destination, solve, *, repetitions, phase) -> dict:
        """Create a native experiment and record graded attempts for each exercise."""
        destination.mkdir(parents=True, exist_ok=False)
        name = phase + "-" + uuid4().hex
        dataset = self._create_dataset(name, exercises)
        examples = list(
            cursor_items(self.client.datasets.list_examples, {"dataset": dataset.id}, "examples")
        )
        by_name = {exercise.slug: exercise for exercise in exercises}
        results, runs = [], []
        for example in examples:
            value = example["input"]
            if isinstance(value, str):
                value = json.loads(value)
            exercise = by_name[value["exercise"]]
            for _ in range(repetitions):
                started = datetime.now(UTC)
                result = attempt(
                    exercise, policy, destination / (exercise.slug + "-" + uuid4().hex), solve
                )
                results.append(result)
                runs.append(
                    {
                        "example_id": example["id"],
                        "output": result,
                        "result.trace.id": result["trace_id"],
                        "result.trace.timestamp": int(started.timestamp() * 1000),
                        "eval.test_fraction.score": result["reward"],
                        "execution_status": "COMPLETED",
                    }
                )
        experiment = self.client.experiments.create(
            name=name,
            dataset=dataset.id,
            experiment_runs=runs,
            task_fields=ExperimentTaskFieldNames(example_id="example_id", output="output"),
        )
        receipt = {
            "experiment_id": experiment.id,
            "dataset_id": dataset.id,
            "trace_project": self.project,
            "space": self.space,
            "attempts": results,
        }
        (destination / "collection.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt

    def chats(self, receipt, policy):
        """Export the experiment model calls with their native task groups and grades."""
        expected = self._await_calls(receipt)
        chats = chat_dataset_from_tracing_provider(
            ArizeAXTracingProvider(
                self.client,
                ModelCallSpec(rendering_options=policy.rendering_options),
                experiment=receipt["experiment_id"],
                dataset=receipt["dataset_id"],
                trace_project=receipt.get("trace_project", self.project),
                space=receipt.get("space", self.space),
                score_name="test_fraction",
            )
        )
        if set(chats["span_id"]) != expected or len(chats) != len(expected):
            raise ValueError("Provider export omitted model calls; no partial PPO batch is allowed")
        return chats

    def _await_calls(self, receipt) -> set[str]:
        """Wait for every completed SDK call to appear in the native span export."""
        expected = {}
        for execution in receipt["attempts"]:
            calls = execution.get("model_call_span_ids")
            if not isinstance(calls, list) or not calls or len(set(calls)) != len(calls):
                raise ValueError("Receipt needs completed SDK call IDs; collect a fresh experiment")
            expected[execution["trace_id"]] = set(calls)
        deadline = time.monotonic() + 90
        visible = set()
        while True:
            for trace_id in expected.keys() - visible:
                rows = cursor_items(
                    self.client.spans.list,
                    {
                        "project": receipt.get("trace_project", self.project),
                        "space": receipt.get("space", self.space),
                        "filter": "context.trace_id = '" + trace_id.replace("'", "''") + "'",
                    },
                    "spans",
                )
                calls = {
                    str(span["SpanId"])
                    for trace in traces_from_spans(rows)
                    for span in trace.spans
                    if span["SpanName"] in MODEL_CALL_SPANS
                    and (
                        span.get("status_code") != "ERROR"
                        or has_completed_output(span["SpanAttributes"])
                    )
                }
                if calls - expected[trace_id]:
                    raise ValueError("Linked trace contains unrecorded SDK model calls")
                if calls == expected[trace_id]:
                    visible.add(trace_id)
            if visible == expected.keys():
                return set().union(*expected.values())
            if time.monotonic() >= deadline:
                raise TimeoutError("AX has not indexed all completed SDK model calls")
            time.sleep(0.5)

    def _create_dataset(self, name, exercises):
        """Create dataset."""
        dataset = self.client.datasets.create(
            name=name,
            space=self.space,
            examples=[
                {"input": {"exercise": e.slug}, "output": {}, "metadata": e.manifest()}
                for e in exercises
            ],
        )
        return dataset
