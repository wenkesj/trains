# Use Phoenix's native dataset, experiment, execution, and evaluation APIs.

import json
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from datasets import Dataset
from phoenix.client import Client
from trains.datasets import chat_dataset_from_tracing_provider
from trains.tracing import ArizePhoenixTracingProvider

from openai_tracing.benchmark import Exercise
from openai_tracing.result import AgentResult
from openai_tracing.sandbox import Sandbox
from openai_tracing.spec import ModelCallSpec
from openai_tracing.training import Policy
from openai_tracing.workflow import attempt


class PhoenixBackend:
    """Collect and export graded coding experiments through the Phoenix client."""

    def __init__(self, client: Client, project: str) -> None:
        """Bind native Phoenix collection to its destination project.

        Args:
            client: Initialized native provider client owned by the application.
            project: Native trace destination project.
        """
        self.client, self.project = client, project

    def collect(
        self,
        exercises: list[Exercise],
        policy: Policy,
        destination: Path,
        solve: Callable[[Sandbox, Policy], Coroutine[Any, Any, AgentResult]],
        *,
        repetitions: int,
        phase: str,
    ) -> dict:
        """Create a native experiment and record graded attempts for each exercise."""
        destination.mkdir(parents=True, exist_ok=False)
        dataset, experiment = self._create_experiment(exercises, policy, phase, repetitions)
        by_name = {exercise.slug: exercise for exercise in exercises}
        results = []
        receipt = {
            "experiment_id": experiment["id"],
            "dataset_id": dataset.id,
            "dataset_version_id": dataset.version_id,
            "attempts": results,
        }
        (destination / "collection.json").write_text(json.dumps(receipt, indent=2) + "\n")
        for example in dataset.examples:
            exercise = by_name[example["input"]["exercise"]]
            for repetition in range(1, repetitions + 1):
                started = datetime.now(UTC)
                result = attempt(
                    exercise, policy, destination / (exercise.slug + "-" + uuid4().hex), solve
                )
                run = self.client.experiments.log_run(
                    experiment_id=experiment["id"],
                    dataset_example_id=example["node_id"],
                    repetition_number=repetition,
                    output=result,
                    start_time=started,
                    end_time=datetime.now(UTC),
                    trace_id=result["trace_id"],
                )
                self.client.experiments.log_evaluation(
                    experiment_run_id=run["id"],
                    name="test_fraction",
                    score=result["reward"],
                    explanation=f"{result['tests'] - result['failed']}/{result['tests']} upstream tests passed",
                )
                results.append(result)
                (destination / "collection.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt

    def chats(self, receipt: dict | str, policy: Policy) -> Dataset:
        """Export the experiment model calls with their native task groups and grades."""
        experiment_id = receipt["experiment_id"] if isinstance(receipt, dict) else receipt
        return chat_dataset_from_tracing_provider(
            ArizePhoenixTracingProvider(
                self.client,
                ModelCallSpec(rendering_options=policy.rendering_options),
                experiment=experiment_id,
                trace_project=self.project,
                score_name="test_fraction",
            )
        )

    def _create_experiment(self, exercises, policy, phase, repetitions):
        """Create experiment."""
        dataset = self.client.datasets.create_dataset(
            name=phase + "-" + uuid4().hex,
            inputs=[{"exercise": exercise.slug} for exercise in exercises],
            outputs=[{} for _ in exercises],
            metadata=[exercise.manifest() for exercise in exercises],
        )
        experiment = self.client.experiments.create(
            dataset_id=dataset.id,
            dataset_version_id=dataset.version_id,
            experiment_name=phase,
            repetitions=repetitions,
            experiment_metadata={
                "model": policy.model,
                "trace_project": self.project,
                "rendering_options": policy.rendering_options,
            },
        )
        return dataset, experiment
