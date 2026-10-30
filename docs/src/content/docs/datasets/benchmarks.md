Import the model calls and grades already saved by a benchmark harness.
Reference answers and judge messages do not become model-generated targets.

| Harness | Native artifacts | Example |
| --- | --- | --- |
| Harbor | Completed job directories and captured calls | [Harbor](#/docs/guides/example-harbor) |
| Harvey LAB | Run config, full transcripts, and judge scores | [Harvey](#/docs/guides/example-harvey) |
| Inspect | `.eval` logs with raw model requests | [Inspect](#/docs/guides/example-inspect) |

## Import a run

Install the optional integration matching the harness, then select its artifacts:

```python
from trains.datasets import (
    chat_dataset_from_harbor_job,
    chat_dataset_from_harvey_run,
    chat_dataset_from_inspect_log,
)

harbor = chat_dataset_from_harbor_job("jobs/completed-job")
harvey = chat_dataset_from_harvey_run("results/run-id")
inspect = chat_dataset_from_inspect_log("logs/evaluation.eval")
```

Harbor accepts full saved conversations or exact captured ATIF model calls.
Keep full histories and disable summarization during collection.

Harvey reconstructs task and system messages with the matching `lab-core` release
and task data. Preserve full transcripts and native scores. Omitted reasoning
cannot be recovered. Completed attempts that exhaust turns or context retain
their captured actions and grades; runs without completion metrics are rejected.

Inspect requires `--log-model-api`. It selects the evaluated model, excludes judge
calls, and resolves native attachments. Save embedded media with the conversations.

## Preserve grading and identities

Repeated attempts share a task group; every completed attempt retains its terminal
grade, including failures. Select the native score explicitly when multiple metrics
exist. Missing inputs, ambiguous grades, or incomplete histories fail validation.

Keep the same dataset version, grader, and model identity when comparing runs.
See [Datum](#/docs/guides/datum) for rendering and training masks.
