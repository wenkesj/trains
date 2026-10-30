from fastapi import APIRouter, Depends, HTTPException, Request

from train.api.dependencies import training
from train.service import TrainingService

router = APIRouter(tags=["Baseten Loops sampling"])


@router.post("/sample", status_code=202)
async def sample(
    body: dict, request: Request, service: TrainingService = Depends(training)
) -> dict:
    """Sample tokens through the native Firetitan-compatible endpoint."""
    if (
        body.get("structured_outputs")
        or body.get("model_input_chunks")
        or body.get("include_routed_experts")
    ):
        raise HTTPException(501, "Only unstructured text sampling is supported locally")
    if body.get("prompt_logprobs", 0) > 0:
        raise HTTPException(501, "Top-k prompt scoring is not supported locally")
    uri = body.get("lora") or request.headers.get("X-Checkpoint-Path")
    sampler = service.create_sampler(uri, body.get("model"))
    params = {
        key: body[key]
        for key in ("max_tokens", "temperature", "seed", "top_k", "top_p", "stop")
        if key in body
    }
    if "stop_token_ids" in body:
        params["stop"] = body["stop_token_ids"]
    result = service.sample(
        {
            "sampling_session_id": sampler,
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": body["prompt"]}]},
            "num_samples": body.get("n", 1),
            "sampling_params": params,
            "prompt_logprobs": "prompt_logprobs" in body,
        },
        key=f"loops:{request.headers['Idempotency-Key']}"
        if "Idempotency-Key" in request.headers
        else None,
        source=body,
    )
    return {"operation_id": result.id, "schema_version": 1}


def as_completions(result: dict, payload: dict) -> dict:
    """Convert a token-sampling result into the completions response schema."""
    prompt = [token for chunk in payload["prompt"]["chunks"] for token in chunk["tokens"]]
    prompt_lps = result.get("prompt_logprobs")
    scores = (
        [
            None if lp is None else {str(token): {"logprob": lp}}
            for token, lp in zip(prompt, prompt_lps, strict=True)
        ]
        if prompt_lps
        else None
    )
    return {
        "object": "text_completion",
        "choices": [
            {
                "index": index,
                "token_ids": sequence["tokens"],
                "finish_reason": sequence["stop_reason"],
                "prompt_token_ids": prompt,
                "prompt_logprobs": scores,
                "logprobs": {
                    "tokens": [f"token_id:{token}" for token in sequence["tokens"]],
                    "token_logprobs": sequence["logprobs"],
                },
            }
            for index, sequence in enumerate(result["sequences"])
        ],
    }
