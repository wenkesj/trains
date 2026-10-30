from collections.abc import Iterator
from typing import Any, Self, cast

from pydantic import BaseModel, ConfigDict

from trains.formats.chat import chat_completion_record
from trains.formats.records import HFRecord
from trains.types import ChatRecord

VERSIONS = {f"ATIF-v1.{minor}" for minor in range(9)}


class ATIFSpec(BaseModel):
    """Replay ATIF's exact captured model calls, with optional full native validation.

    Supply validation_class=atif.Trajectory for full schema validation. Native
    summaries cannot recover hidden prompts; those require a custom trace spec.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    trajectory: dict | None = None
    validation_class: Any = None

    @classmethod
    def from_trajectory(cls, trajectory: Any) -> Self:
        """Create an ATIF spec from a parsed trajectory."""
        value = (
            trajectory.to_json_dict(exclude_none=True)
            if hasattr(trajectory, "to_json_dict")
            else trajectory.model_dump(mode="json", exclude_none=True)
        )
        return cls(trajectory=value)

    def iter_chats(
        self,
        trajectory: dict,
        *,
        reward: float | None = None,
        group_id: str | None = None,
    ) -> Iterator[ChatRecord]:
        """Yield complete captured ATIF calls with trajectory provenance."""
        if self.validation_class is not None:
            self.validation_class.model_validate(trajectory)
        model = trajectory.get("agent", {}).get("model_name")
        generations = _captured_generations(trajectory, model)
        if not generations:
            raise ValueError("ATIF trajectory contains no trainable model calls")
        for generation in generations:
            record = cast(ChatRecord, dict(generation))
            record.update(
                {
                    "trace_id": trajectory.get("session_id"),
                    "model_id": model,
                    "reward": reward,
                    "group_id": group_id,
                }
            )
            yield record

    def to_chats(self) -> list[ChatRecord]:
        """Materialize complete captured ATIF calls as chat records."""
        if self.trajectory is None:
            raise ValueError("Use ATIFSpec.from_trajectory first")
        return list(self.iter_chats(self.trajectory))


def _captured_generations(payload: dict, model_id: str | None) -> list[HFRecord]:
    """Replay exact captured calls, never inferred prompts from summarized ATIF steps.

    Per-agent step.extra.trains contains the actual OpenAI chat request and response.
    Copied context and deterministic dispatches are not training targets. This
    converter handles text, single-trajectory ATIF v1.0-v1.8; continuation/subagent
    graphs and aggregated model steps require a dedicated converter.
    """
    if payload.get("schema_version") not in VERSIONS:
        raise ValueError("Unsupported ATIF schema version")
    if payload.get("continued_trajectory_ref") or payload.get("subagent_trajectories"):
        raise ValueError("ATIF continuation/subagent graphs need explicit trajectory selection")
    steps = payload.get("steps") or []
    if [step.get("step_id") for step in steps] != list(range(1, len(steps) + 1)):
        raise ValueError("ATIF step IDs must be sequential")
    values = []
    for step in steps:
        if (
            step.get("source") != "agent"
            or step.get("is_copied_context")
            or step.get("llm_call_count") == 0
        ):
            continue
        if step.get("llm_call_count") not in {None, 1}:
            raise ValueError("Aggregated ATIF steps cannot reconstruct individual model calls")
        model = step.get("model_name") or payload["agent"].get("model_name") or model_id
        if not isinstance(model, str) or not model:
            raise ValueError("ATIF model turns need an explicit model ID")
        if model != model_id:
            continue
        captured = (step.get("extra") or {}).get("trains") or {}
        if not captured.get("request") or not captured.get("response"):
            raise ValueError(
                "ATIF turn lacks exact model request/response capture in step.extra.trains"
            )
        values.extend(_captured_generation(captured, model))
    return values


def _captured_generation(captured: dict, model: str) -> list[HFRecord]:
    """Convert exact ATIF request and response payloads into an HF record."""
    request, response = (captured["request"], captured["response"])
    if request.get("model") != model:
        raise ValueError("Captured request does not match the ATIF model ID")
    choices = response.get("choices") or []
    if len(choices) != 1 or choices[0].get("finish_reason") not in {"stop", "tool_calls"}:
        raise ValueError("Expected one complete model response")
    message = choices[0]["message"]
    if message.get("tool_calls") and (not request.get("tools")):
        raise ValueError("Tool-calling ATIF turns need the actual tool schemas")
    return [chat_completion_record(request, response)]
