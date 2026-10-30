import pytest
from pydantic import ValidationError

from train.api.routes.inference import _bounded_output
from train.serving import Serving


@pytest.mark.parametrize(
    ("path", "field"),
    [
        ("responses", "max_output_tokens"),
        ("messages", "max_tokens"),
        ("completions", "max_tokens"),
        ("chat/completions", "max_tokens"),
        ("chat/completions", "max_completion_tokens"),
    ],
)
def test_output_ceiling_keeps_prompt_tools_and_smaller_budgets(path, field):
    body = {"input": "long prompt", "tools": [{"type": "function"}], field: 64000}
    result = _bounded_output(body, path, 4096)
    assert result == {**body, field: 4096}
    assert body[field] == 64000
    assert _bounded_output({**body, field: 128}, path, 4096)[field] == 128
    assert _bounded_output(body, path, None) is body


def test_default_output_limit_does_not_affect_token_counting():
    body = {"messages": [{"role": "user", "content": "keep every token"}]}
    assert _bounded_output(body, "responses", 4096) == {**body, "max_output_tokens": 4096}
    assert _bounded_output(body, "messages/count_tokens", 4096) is body


@pytest.mark.parametrize("invalid", [0, -1, True, "bad"])
def test_invalid_client_budget_is_left_for_backend_validation(invalid):
    body = {"max_output_tokens": invalid}
    assert _bounded_output(body, "responses", 4096) == body


def test_serving_rejects_nonpositive_ceiling():
    with pytest.raises(ValidationError):
        Serving(max_output_tokens=0)
