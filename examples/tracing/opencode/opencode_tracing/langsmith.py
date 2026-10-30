# Native LangSmith dataset examples, experiment projects, runs, and feedback.

import json
import time
from datetime import UTC, datetime
from uuid import uuid4

from langsmith.run_trees import RunTree
from recipes.tracing import has_completed_output, langsmith_run_id
from trains.datasets import chat_dataset_from_tracing_provider
from trains.tracing import LangSmithTracingProvider

from opencode_tracing.spec import MODEL_CALL_SPANS, ModelCallSpec
from opencode_tracing.workflow import attempt


class LangSmithBackend:
    """Collect and export graded coding experiments through the LangSmith client."""

    def __init__(self, client, *, project: str) -> None:
        """Bind native LangSmith collection to its destination project.

        Args:
            client: Initialized native provider client owned by the application.
            project: Native trace destination project.
        """
        self.client, self.project = client, project

    def collect(self, exercises, policy, destination, solve, *, repetitions, phase) -> dict:
        """Create a native experiment and record graded attempts for each exercise."""
        destination.mkdir(parents=True, exist_ok=False)
        name = phase + "-" + uuid4().hex
        examples, experiment = self._create_experiment(name, exercises, policy, repetitions)
        by_name = {exercise.slug: exercise for exercise in exercises}
        results = []
        for example in examples:
            exercise = by_name[example.inputs["exercise"]]
            for _ in range(repetitions):
                started = datetime.now(UTC)
                result = attempt(
                    exercise, policy, destination / (exercise.slug + "-" + uuid4().hex), solve
                )
                identifier = uuid4()
                self.client.create_run(
                    name="exercism.attempt",
                    inputs=example.inputs,
                    run_type="chain",
                    id=identifier,
                    trace_id=identifier,
                    dotted_order=RunTree(
                        name="exercism.attempt", id=identifier, start_time=started
                    ).dotted_order,
                    extra={"metadata": {"OTEL_TRACE_ID": result["trace_id"]}},
                    project_name=name,
                    start_time=started,
                    end_time=datetime.now(UTC),
                    outputs=result,
                    reference_example_id=example.id,
                )
                self.client.create_feedback(
                    run_id=identifier,
                    key="test_fraction",
                    score=result["reward"],
                    session_id=experiment.id,
                )
                results.append(result)
        self.client.flush()
        receipt = {
            "experiment_id": name,
            "dataset": name,
            "trace_project": self.project,
            "attempts": results,
        }
        (destination / "collection.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt

    def chats(self, receipt, policy):
        """Export the experiment model calls with their native task groups and grades."""
        self._await_experiment(receipt)
        chats = chat_dataset_from_tracing_provider(
            LangSmithTracingProvider(
                self.client,
                ModelCallSpec(rendering_options=policy.rendering_options),
                experiment=receipt["experiment_id"],
                dataset=receipt["dataset"],
                trace_project=receipt.get("trace_project", self.project),
                score_name="test_fraction",
            )
        )
        if len(set(chats["trace_id"])) != len(receipt["attempts"]):
            raise ValueError("Provider export omitted an attempt; no partial PPO batch is allowed")
        expected_calls = {
            langsmith_run_id(span_id)
            for attempt in receipt["attempts"]
            for span_id in attempt["model_call_span_ids"]
        }
        if len(chats) != len(expected_calls):
            raise ValueError("Provider export omitted model calls; no partial PPO batch is allowed")
        if set(chats["span_id"]) != expected_calls:
            raise ValueError(
                "Provider export changed model call identities; no partial PPO batch is allowed"
            )
        return chats

    def _await_experiment(self, receipt) -> None:
        """Wait for every native experiment attempt and evaluator score to be indexed."""
        expected = {attempt["trace_id"] for attempt in receipt["attempts"]}
        expected_calls = {}
        for execution in receipt["attempts"]:
            calls = execution.get("model_call_span_ids")
            if not isinstance(calls, list) or not calls or len(set(calls)) != len(calls):
                raise ValueError("Receipt needs completed SDK call IDs; collect a fresh experiment")
            expected_calls[execution["trace_id"]] = {langsmith_run_id(call) for call in calls}
        deadline = time.monotonic() + 90
        visible = set()
        while True:
            runs = list(self.client.list_runs(project_name=receipt["experiment_id"], is_root=True))
            present = {(run.extra or {}).get("metadata", {}).get("OTEL_TRACE_ID") for run in runs}
            if present == expected and len(runs) == len(expected):
                identifiers = {str(run.id) for run in runs}
                scores = self.client.list_feedback(
                    run_ids=[run.id for run in runs], feedback_key=["test_fraction"]
                )
                graded = {str(score.run_id) for score in scores if score.score is not None}
                if graded == identifiers:
                    for trace_id in expected - visible:
                        linked = self.client.list_runs(
                            project_name=receipt.get("trace_project", self.project),
                            run_type="llm",
                            filter='and(eq(metadata_key,"OTEL_TRACE_ID"),eq(metadata_value,'
                            + json.dumps(trace_id)
                            + "))",
                            select=["id", "name", "trace_id", "extra", "error", "outputs"],
                        )
                        calls = {
                            str(run.id)
                            for run in linked
                            if run.name in MODEL_CALL_SPANS
                            and (
                                not run.error or has_completed_output({"output.value": run.outputs})
                            )
                        }
                        if calls - expected_calls[trace_id]:
                            raise ValueError("Linked trace contains unrecorded SDK model calls")
                        if calls == expected_calls[trace_id]:
                            visible.add(trace_id)
                    if visible == expected:
                        return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "LangSmith has not indexed all attempts, grades, and linked model traces"
                )
            time.sleep(0.5)

    def _create_experiment(self, name, exercises, policy, repetitions):
        """Create experiment."""
        dataset = self.client.create_dataset(dataset_name=name)
        examples = [
            self.client.create_example(
                dataset_id=dataset.id,
                inputs={"exercise": e.slug},
                metadata=e.manifest(),
            )
            for e in exercises
        ]
        version = max(example.modified_at or example.created_at for example in examples)
        experiment = self.client.create_project(
            project_name=name,
            reference_dataset_id=dataset.id,
            metadata={"dataset_version": version.isoformat(), "model": policy.model},
            num_repetitions=repetitions,
        )
        return examples, experiment
