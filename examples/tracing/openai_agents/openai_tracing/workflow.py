# One real exercise attempt, isolated execution, and upstream test grading.

import asyncio
import json
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from opentelemetry import trace

from openai_tracing.benchmark import Exercise
from openai_tracing.instrumentation import model_calls
from openai_tracing.result import AgentResult
from openai_tracing.sandbox import Sandbox
from openai_tracing.training import Policy


def attempt(
    exercise: Exercise,
    policy: Policy,
    destination: Path,
    solve: Callable[[Sandbox, Policy], Coroutine[Any, Any, AgentResult]],
) -> dict:
    """Solve one isolated exercise, grade its files, and save the trace receipt."""
    destination.mkdir(parents=True, exist_ok=False)
    candidate, verifier = destination / "candidate", destination / "verifier"
    exercise.candidate(candidate)
    tracer = trace.get_tracer("trains.tracing.examples")
    with tracer.start_as_current_span(
        "exercism.attempt",
        attributes={"openinference.span.kind": "AGENT", "session.id": destination.name},
    ) as root:
        with Sandbox(candidate) as sandbox:
            response = asyncio.run(solve(sandbox, policy))
        exercise.verifier(candidate, verifier)
        with Sandbox(verifier) as sandbox:
            grade = sandbox.grade()
        result = {
            **exercise.manifest(),
            **grade,
            "trace_id": f"{root.get_span_context().trace_id:032x}",
            "model_id": policy.model,
            "response": response.response,
            "stop_reason": response.stop_reason,
        }
    result["model_call_span_ids"] = model_calls.take(result["trace_id"])
    (destination / "attempt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
