from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from phoenix.client.resources.datasets import Dataset

from dataset_fixtures import span_frame
from trains.tracing.providers.experiments import ExperimentRun


@pytest.fixture
def native_run():
    return ExperimentRun(
        id="run",
        trace_id="trace-native",
        experiment_id="experiment",
        example_id="task-1",
        dataset_id="dataset",
        dataset_version_id="version",
        project="project",
        evaluations=[{"name": "quality", "score": 0.75}],
    )


@pytest.fixture
def phoenix_async_client():
    return SimpleNamespace(
        experiments=SimpleNamespace(
            get_experiment=AsyncMock(
                return_value={
                    "experiment_id": "experiment",
                    "dataset_id": "dataset",
                    "dataset_version_id": "version",
                    "project_name": "project",
                    "task_runs": [
                        {"id": "run", "trace_id": "trace-native", "dataset_example_id": "example"}
                    ],
                    "evaluation_runs": [],
                }
            )
        ),
        datasets=SimpleNamespace(
            get_dataset=AsyncMock(
                return_value=Dataset(
                    {"id": "dataset", "name": "dataset"},
                    {"version_id": "version", "examples": [{"id": "example"}]},
                )
            )
        ),
        spans=SimpleNamespace(get_spans_dataframe=AsyncMock(return_value=span_frame())),
    )
