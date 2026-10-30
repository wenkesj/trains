from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import httpx
from transformers import AutoConfig

from train.models import ModelInput, ModelSpec, SamplingParams
from train.serving import locked

if TYPE_CHECKING:
    from train.service import TrainingService


def supports(body: dict) -> bool:
    """Select native text requests whose raw probability semantics are preserved."""
    prompt = ModelInput.model_validate(body["prompt"])
    params = SamplingParams.model_validate(body.get("sampling_params", {}))
    stops = params.stop if params.stop is not None else []
    return (
        all(chunk.type == "encoded_text" for chunk in prompt.chunks)
        and not body.get("target_prompt_logprobs")
        and not isinstance(stops, str)
        and all(isinstance(stop, int) for stop in stops)
        and (
            params.temperature == 0
            or (params.temperature == 1 and params.top_p == 1 and params.top_k <= 0)
        )
    )


def sample(service: TrainingService, body: dict) -> dict:
    """Sample exact token inputs from a pinned native SGLang checkpoint.

    Args:
        service: Native training service with a configured SGLang backend.
        body: Native sampling request including prompt, checkpoint, and settings.

    Returns:
        Untrimmed token IDs, raw model log probabilities, and stop metadata.
    """
    params = SamplingParams.model_validate(body.get("sampling_params", {}))
    prompt = ModelInput.model_validate(body["prompt"]).tokens()
    uri = body["model_path"]
    base = uri.removeprefix("base://") if uri.startswith("base://") else None
    spec = (
        ModelSpec(base_model=base, base_path=service.models[base], rank=1)
        if base is not None
        else ModelSpec.load(service.checkpoint(uri) / "model.json")
    )
    config = AutoConfig.from_pretrained(
        spec.base_path, local_files_only=True, trust_remote_code=service.runtime.trust_remote_code
    ).get_text_config()
    count, topk = body.get("num_samples", 1), body.get("topk_sample_logprobs", 0)
    context = getattr(config, "max_position_embeddings", 4096)
    if not 0 <= count <= 16 or not prompt or len(prompt) + (params.max_tokens or 16) > context:
        raise ValueError("Invalid sample count, empty prompt, or context limit exceeded")
    if any(token < 0 or token >= config.vocab_size for token in prompt):
        raise ValueError("Prompt token is out of vocabulary")
    if topk < 0 or topk > config.vocab_size:
        raise ValueError("Invalid top-k logprob count")
    if count == 0 and not body.get("prompt_logprobs"):
        return {"sequences": [], "prompt_logprobs": None}
    request = _request(body, prompt, params, count, topk)
    serving = service.runtime.serving
    with (
        locked(service.root),
        httpx.Client(headers=serving.headers, timeout=serving.timeout) as client,
    ):
        service.deployments.check_backend(client, spec.base_model)
        selected = service.deployments.ensure_loaded(
            client, None if base is not None else uri, spec.base_model
        )
        if selected != (serving.served_model or spec.base_model):
            request["lora_path"] = selected.rpartition(":")[2]
        response = client.post(serving.endpoint("/generate"), json=request)
        response.raise_for_status()
        outputs = response.json()
    outputs = outputs if isinstance(outputs, list) else [outputs]
    if len(outputs) != max(count, 1):
        raise ValueError("Native sampler returned an incomplete sequence batch")
    sequences = [_sequence(output, prompt, params, topk) for output in outputs]
    prompt_logprobs = _prompt_scores(outputs[0], prompt) if body.get("prompt_logprobs") else None
    return {"sequences": sequences if count else [], "prompt_logprobs": prompt_logprobs}


def _prompt_scores(output: dict, prompt: list[int]) -> list[float | None]:
    """Require complete causal prompt scores aligned with the original input."""
    values = output["meta_info"]["input_token_logprobs"]
    if len(values) != len(prompt) or [value[1] for value in values] != prompt:
        raise ValueError("Native prompt scores do not match the requested input tokens")
    scores = [value[0] for value in values]
    if scores[0] is not None or not all(
        value is not None and math.isfinite(value) for value in scores[1:]
    ):
        raise ValueError("Native prompt scores are incomplete or nonfinite")
    return scores


def _request(body: dict, prompt: list[int], params: SamplingParams, count: int, topk: int) -> dict:
    """Retain original token IDs, EOS tokens, and native sampling settings."""
    return {
        "input_ids": prompt,
        "return_prompt_token_ids": True,
        "return_logprob": True,
        "logprob_start_len": 0 if body.get("prompt_logprobs") else -1,
        "top_logprobs_num": topk,
        "sampling_params": {
            "max_new_tokens": (params.max_tokens or 16) if count else 0,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k if params.top_k > 0 else -1,
            "sampling_seed": params.seed,
            "stop_token_ids": params.stop,
            "no_stop_trim": True,
            "skip_special_tokens": False,
            "n": max(count, 1),
        },
    }


def _sequence(output: dict, prompt: list[int], params: SamplingParams, topk: int) -> dict[str, Any]:
    """Reject altered prompts and token/probability misalignment before export."""
    if output.get("prompt_token_ids") != prompt:
        raise ValueError("Native sampler changed the requested input token IDs")
    metadata = output["meta_info"]
    reason = metadata["finish_reason"]["type"]
    if reason not in {"stop", "length"}:
        raise ValueError(f"Native sampler did not complete: {reason}")
    tokens = output["output_ids"]
    values = metadata["output_token_logprobs"]
    if len(values) != len(tokens) or [value[1] for value in values] != tokens:
        raise ValueError("Native action tokens and log probabilities are misaligned")
    logprobs = [value[0] for value in values]
    if not all(math.isfinite(value) for value in logprobs):
        raise ValueError("Native action log probabilities are nonfinite")
    alternatives = None
    if topk:
        alternatives = metadata["output_top_logprobs"]
        if len(alternatives) != len(tokens) or any(len(row) != topk for row in alternatives):
            raise ValueError("Native top-k scores are incomplete")
        alternatives = [[(value[1], value[0]) for value in row] for row in alternatives]
    return {
        "tokens": tokens,
        "logprobs": logprobs,
        "sampling_logprobs": logprobs if params.temperature else [0.0] * len(tokens),
        "stop_reason": reason,
        "topk_logprobs": alternatives,
    }
