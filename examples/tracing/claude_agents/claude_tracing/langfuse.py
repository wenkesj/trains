# Native Langfuse datasets, experiments, evaluations, and OpenInference export.

import asyncio
import json
import time
from datetime import UTC, datetime
from uuid import uuid4

from datasets import Dataset
from langfuse import Langfuse
from langfuse.experiment import Evaluation
from trains.datasets import chat_dataset_from_tracing_provider
from trains.tracing import LangfuseTracingProvider
from trains.tracing.providers.native import native_mapping

from claude_tracing.spec import ModelCallSpec
from claude_tracing.training import Policy
from claude_tracing.workflow import attempt


class LangfuseBackend:
    """Collect and export graded coding experiments through the Langfuse client."""

    def __init__(self, client: Langfuse) -> None:
        """Use the initialized Langfuse client for native experiment collection.

        Args:
            client: Initialized native provider client owned by the application.
        """
        self.client = client

    def collect(self, exercises, policy, destination, solve, *, repetitions, phase) -> dict:
        """Create a native experiment and record graded attempts for each exercise."""
        destination.mkdir(parents=True, exist_ok=False)
        name = phase + "-" + uuid4().hex
        self.client.create_dataset(name=name)
        for exercise in exercises:
            self.client.create_dataset_item(
                dataset_name=name, input={"exercise": exercise.slug}, metadata=exercise.manifest()
            )
        dataset = self.client.get_dataset(name)
        by_name = {exercise.slug: exercise for exercise in exercises}
        started = datetime.now(UTC)
        attempts = []

        async def task(*, item, **kwargs):
            result = await asyncio.to_thread(
                attempt,
                by_name[item.input["exercise"]],
                policy,
                destination / (item.input["exercise"] + "-" + uuid4().hex),
                solve,
            )
            attempts.append(result)
            return result

        result = self.client.run_experiment(
            name=phase,
            run_name=name,
            data=[item for item in dataset.items for _ in range(repetitions)],
            task=task,
            evaluators=[_evaluate],
            max_concurrency=1,
            metadata={"model": policy.model, "benchmark": "exercism-python"},
        )
        self.client.flush()
        if len(result.item_results) != len(exercises) * repetitions:
            raise RuntimeError(
                "Langfuse experiment contains task errors; inspect the native experiment"
            )
        receipt = {
            "experiment_id": result.experiment_id,
            "dataset": name,
            "from_start_time": started.isoformat(),
            "attempts": attempts,
        }
        (destination / "collection.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt

    def chats(self, receipt: dict, policy: Policy) -> Dataset:
        """Export the experiment model calls with their native task groups and grades."""
        provider = LangfuseTracingProvider(
            self.client,
            ModelCallSpec(rendering_options=policy.rendering_options),
            experiment=receipt["experiment_id"],
            dataset=receipt["dataset"],
            from_start_time=datetime.fromisoformat(receipt["from_start_time"]),
            score_name="test_fraction",
        )
        deadline = time.monotonic() + 120
        while True:
            experiments = native_mapping(
                self.client.api.experiments.list(
                    id=receipt["experiment_id"],
                    from_start_time=datetime.fromisoformat(receipt["from_start_time"]),
                    fields="core",
                )
            )["data"]
            if not experiments:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Langfuse has not indexed the experiment")
                time.sleep(1)
                continue
            runs = list(provider.iter_experiment_runs())
            if len(runs) == len(receipt["attempts"]) and all(
                len(run.evaluations) == 1 for run in runs
            ):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Langfuse has not indexed every experiment execution and score")
            time.sleep(1)
        chats = chat_dataset_from_tracing_provider(provider)
        expected_traces = {attempt["trace_id"] for attempt in receipt["attempts"]}
        expected_calls = {
            span_id for attempt in receipt["attempts"] for span_id in attempt["model_call_span_ids"]
        }
        if set(chats["trace_id"]) != expected_traces:
            raise ValueError("Provider export omitted an attempt; no partial PPO batch is allowed")
        if len(chats) != len(expected_calls) or set(chats["span_id"]) != expected_calls:
            raise ValueError("Provider export differs from the exact recorded SDK model-call IDs")
        return chats


def _evaluate(*, output, **kwargs):
    """Return the native test-fraction evaluation for an attempt."""
    return Evaluation(name="test_fraction", value=output["reward"])
