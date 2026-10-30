import asyncio

import zstandard
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from tinker.types import ClientConfigResponse

from train.api.dependencies import training
from train.models import ModelSpec
from train.service import TrainingService
from train.wire import decode_forward, encode_forward, encode_sample

router = APIRouter(prefix="/api/v1", tags=["Tinker / Spindle / Firetitan"])


@router.get("/healthz")
async def health() -> dict:
    """Report training-service readiness."""
    return {"status": "ok"}


@router.get("/get_server_capabilities")
async def capabilities(service: TrainingService = Depends(training)) -> dict:
    """Report registered models and supported native training operations."""
    return {"supported_models": [{"model_name": name} for name in service.models]}


@router.post("/client/config")
async def config() -> dict:
    """Return the native client configuration."""
    return {
        **ClientConfigResponse().model_dump(),
        "parallel_fwdbwd_chunks": False,
        "use_pyqwest_transport": False,
        "proto_write_fwdbwd": False,
        "proto_read_fwdbwd": False,
        "proto_compress_fwdbwd": False,
    }


@router.post("/client/dynamic_config")
async def dynamic_config() -> dict:
    """Return polling settings for native asynchronous operations."""
    return {}


@router.post("/create_session")
async def create_session(service: TrainingService = Depends(training)) -> dict:
    """Create a native Tinker training session."""
    return {"session_id": service.create_session(), "type": "create_session"}


@router.post("/sessions/{session_id}/finish", status_code=204)
async def finish(session_id: str, service: TrainingService = Depends(training)) -> Response:
    """Close a native Tinker session."""
    service.sessions.discard(session_id)
    runs = [run for run in service.runs.values() if run.session_id == session_id]
    await asyncio.gather(*(run.task for run in runs if run.task is not None))
    for run in runs:
        run.engine = None
        for operation in run.pending.values():
            operation.error = "Session closed before this sequence could execute"
            operation.done = True
        run.pending.clear()
    return Response(status_code=204)


@router.post("/session_heartbeat")
@router.post("/telemetry")
async def heartbeat() -> dict:
    """Keep a native client session active."""
    return {"status": "accepted"}


@router.post("/retrieve_future")
async def retrieve(request: Request, service: TrainingService = Depends(training)) -> Response:
    """Poll a native Tinker asynchronous operation."""
    body = await request.json()
    operation = service.operations[body["request_id"]]
    if not operation.done:
        return JSONResponse({"type": "try_again", "queue_state": "active"}, status_code=408)
    if operation.error:
        return JSONResponse({"error": operation.error, "category": "user"})
    result = operation.result or {}
    if "application/x-protobuf" in request.headers.get("accept", ""):
        encoder = {
            "forward": encode_forward,
            "forward_backward": encode_forward,
            "sample": encode_sample,
        }.get(operation.kind)
        if encoder:
            return Response(encoder(result), media_type="application/x-protobuf")
    return JSONResponse(result)


@router.post("/{operation}")
async def submit(
    operation: str, request: Request, service: TrainingService = Depends(training)
) -> dict:
    """Submit a native Tinker training or sampling operation."""
    if request.headers.get("content-type", "").startswith("application/x-protobuf"):
        raw = await request.body()
        if request.headers.get("content-encoding") == "zstd":
            raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=64 * 1024 * 1024)
        body = decode_forward(raw)
    else:
        body = await request.json()
    if operation == "create_model" and (
        request.headers.get("user-agent", "").startswith("AsyncTrainingHTTPClient/")
        or request.headers.get("X-Trains-Provider") == "fireworks"
    ):
        body["_protocol"] = "fireworks"
    return _dispatch(service, operation, body)


def _dispatch(service: TrainingService, operation: str, body: dict) -> dict:
    """Translate a native request into a local training-service operation."""
    if operation in {"weights_info", "get_info"}:
        return _describe(service, operation, body)
    if operation == "create_sampling_session":
        identifier = service.create_sampler(body.get("model_path"), body.get("base_model"))
        return {"sampling_session_id": identifier, "type": operation}
    if operation == "create_model":
        result = service.create_model(body)
    elif operation == "asample":
        result = service.sample(body)
    elif operation in {
        "forward_backward",
        "forward",
        "optim_step",
        "save_weights",
        "save_weights_for_sampler",
        "load_weights",
    }:
        if operation == "forward" or body.get("forward_only"):
            operation = "forward"
        if "forward_input" in body:
            body["forward_backward_input"] = body.pop("forward_input")
        result = service.submit(body["model_id"], body.get("seq_id"), operation, body)
    else:
        raise HTTPException(501, f"Unsupported local training operation: {operation}")
    future: dict = {"request_id": result.id, "type": "untyped_api_future"}
    if operation == "asample":
        future["sample_sequence_ids"] = [
            f"{result.id}:{index}" for index in range(body.get("num_samples", 1))
        ]
    return future


def _describe(service: TrainingService, operation: str, body: dict) -> dict:
    """Report the registered trainer model configuration."""
    if operation == "weights_info":
        spec = ModelSpec.load(
            service.checkpoint(body.get("tinker_path") or body["path"]) / "model.json"
        )
        return {
            "base_model": spec.base_model,
            "is_lora": True,
            "lora_rank": spec.rank,
            "train_mlp": spec.train_mlp,
            "train_attn": spec.train_attn,
            "train_unembed": spec.train_unembed,
        }
    if operation == "get_info":
        run = service.runs[body["model_id"]]
        return {
            "model_id": run.id,
            "is_lora": True,
            "lora_rank": run.spec.rank,
            "model_data": {
                "model_name": run.spec.base_model,
                "tokenizer_id": str(run.spec.base_path),
            },
        }
    raise HTTPException(501, "Unsupported metadata operation")
