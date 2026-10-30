# Expose the pinned Chartography task with a fixed local Omni judge.

from chartography.task import chartography as _chartography
from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from recipes.models import DEFAULT_MODEL

from chartqa import load_charts, relaxed_accuracy

DEFAULT_JUDGE_MODEL = f"openai-api/judge/{DEFAULT_MODEL}"


@task
def chartography(
    hf_repo: str | None = None,
    split: str = "test",
    task_pack: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    model_family: str | None = "lfm3",
    judge_model_family: str | None = "lfm3",
) -> Task:
    """Load original chart tasks with disjoint train and evaluation slices.

    Args:
        hf_repo: Native Hugging Face dataset identity.
        split: Native dataset split.
        task_pack: Alternative local upstream task pack.
        limit: Number of charts in this slice.
        offset: First chart index, used to reserve held-out samples.
        judge_model: Fixed judge identity, independent of the trained sampler.
        model_family: Inspect capability family for renamed sampler identities.
        judge_model_family: Inspect capability family for the fixed judge.

    Returns:
        The upstream Inspect task with the selected chart slice.
    """
    if offset < 0 or (limit is not None and limit < 1):
        raise ValueError("Use a nonnegative offset and a positive chart limit")
    result = _chartography(
        hf_repo=hf_repo,
        split=split,
        task_pack=task_pack,
        limit=offset + limit if limit is not None else None,
        judge_model=judge_model,
        model_family=model_family,
        judge_model_family=judge_model_family,
    )
    result.dataset = result.dataset[offset:]
    if not len(result.dataset):
        raise ValueError("The selected chart slice is empty")
    return result


@task
def chartqa(split: str = "train", limit: int = 4, offset: int = 0) -> Task:
    """Evaluate public chart questions with original labels and deterministic scoring.

    Args:
        split: Use train for optimization and test for held-out evaluation.
        limit: Number of original chart questions.
        offset: First row in the selected split.

    Returns:
        A native Inspect vision task without hosted judge requirements.
    """
    return Task(
        dataset=load_charts(split, limit=limit, offset=offset),
        solver=generate(),
        scorer=relaxed_accuracy(),
        config=GenerateConfig(max_tokens=256),
    )
