import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from train.api.dependencies import training
from train.api.routes.inference import proxy
from train.service import TrainingService

router = APIRouter(tags=["Firetitan token sampling"])


@router.post("/inference/v1/completions")
async def completions(
    body: dict, request: Request, service: TrainingService = Depends(training)
) -> Response:
    """Execute a native Fireworks token completion request."""
    if (
        body.get("images")
        or body.get("include_routing_matrix")
        or body.get("top_sampling_logprobs")
    ):
        raise HTTPException(501, "Only text completions without router replay are supported")
    model = body["model"]
    if model.startswith("accounts/"):
        return await proxy(body, "completions", service)
    sampler = service.create_sampler(None if model in service.models else model, model)
    prompt = body["prompt"]
    if not isinstance(prompt, list) or any(type(token) is not int for token in prompt):
        raise HTTPException(422, "Local completions require a list of prompt token IDs")
    params = {
        key: body[key]
        for key in ("max_tokens", "temperature", "seed", "top_k", "top_p", "stop")
        if key in body
    }
    operation = service.sample(
        {
            "sampling_session_id": sampler,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": prompt}]},
            "sampling_params": params,
            "num_samples": body.get("n", 1),
            "prompt_logprobs": bool(body.get("echo")),
        },
        key=f"fireworks:{request.headers['Idempotency-Key']}"
        if "Idempotency-Key" in request.headers
        else None,
        source=body,
    )
    while not operation.done:
        await asyncio.sleep(0.01)
    if operation.error:
        raise HTTPException(422, operation.error)
    result = _completion_response(operation.result or {}, prompt, bool(body.get("echo")))
    if body.get("stream"):
        return Response(
            f"data: {json.dumps(result)}\n\ndata: [DONE]\n\n", media_type="text/event-stream"
        )
    return JSONResponse(result)


def _completion_response(result: dict, prompt: list[int], echo: bool) -> dict:
    """Encode sampled tokens as a Fireworks completion response."""
    choices = []
    for index, sequence in enumerate(result["sequences"]):
        tokens = (prompt if echo else []) + sequence["tokens"]
        prompt_lps = [value or 0.0 for value in result.get("prompt_logprobs") or []] if echo else []
        raw = prompt_lps + sequence["logprobs"]
        sampled = prompt_lps + sequence["sampling_logprobs"]
        choices.append(
            {
                "index": index,
                "text": "",
                "finish_reason": sequence["stop_reason"],
                "raw_output": {"completion_token_ids": tokens},
                "prompt_token_ids": prompt,
                "logprobs": {
                    "content": [
                        {"token": f"token_id:{token}", "logprob": lp, "sampling_logprob": slp}
                        for token, lp, slp in zip(tokens, raw, sampled, strict=True)
                    ]
                },
            }
        )
    completion_tokens = sum(len(sequence["tokens"]) for sequence in result["sequences"])
    return {
        "id": "local-completion",
        "object": "text_completion",
        "choices": choices,
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": completion_tokens,
            "total_tokens": len(prompt) + completion_tokens,
        },
    }
