from uuid import NAMESPACE_URL, uuid5

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from train.api.dependencies import training
from train.api.routes.completions import as_completions
from train.checkpoints import Checkpoint
from train.models import ModelSpec
from train.service import TrainingService

router = APIRouter(tags=["Baseten Loops"])


@router.get("/v1/loops/capabilities")
async def capabilities(service: TrainingService = Depends(training)) -> dict:
    """Report the local Loops model and loss capabilities."""
    return {"supported_models": [{"model_name": name} for name in service.models]}


@router.post("/v1/loops/sessions")
async def create_session(service: TrainingService = Depends(training)) -> dict:
    """Create a native Loops training session."""
    return {"session": {"id": service.create_session()}}


@router.get("/v1/loops/checkpoints")
async def checkpoints(
    checkpoint_path: str | None = None,
    run_id: str | None = None,
    name: str | None = None,
    target: str | None = None,
    service: TrainingService = Depends(training),
) -> dict:
    """List native Loops checkpoint identities."""
    if checkpoint_path:
        records = [
            Checkpoint(uri=checkpoint_path, directory=service.checkpoint(checkpoint_path), step=0)
        ]
    else:
        records = Checkpoint.listing(service.root, service.models, run_id)
    rows = [record.metadata() for record in records]
    return {
        "checkpoints": [
            row
            for row in rows
            if (name is None or row["checkpoint_id"] == name)
            and (target is None or row["target"] == target)
        ]
    }


@router.post("/v1/loops/trainers")
async def create_trainer(
    body: dict, request: Request, service: TrainingService = Depends(training)
) -> dict:
    """Create a local LoRA trainer for a Loops request."""
    if body.get("replicas", 1) != 1:
        raise HTTPException(501, "Local training supports one replica")
    config = {"rank": body["lora_rank"], "seed": body.get("seed"), "train_unembed": False}
    if body.get("path"):
        spec = ModelSpec.load(service.checkpoint(body["path"]) / "model.json")
        if spec.rank != body["lora_rank"] or spec.base_model != body["base_model"]:
            raise ValueError("Checkpoint configuration does not match trainer")
        config = spec.model_dump(exclude={"base_model", "base_path"})
    operation = service.create_model(
        {**body, "model_seq_id": len(service.runs), "_protocol": "tinker", "lora_config": config}
    )
    run = next(run for run in service.runs.values() if run.history[0] is operation)
    return {
        "run": {
            "id": run.id,
            "base_model": run.spec.base_model,
            "base_url": str(request.base_url).rstrip("/"),
            "deployment_id": None,
        }
    }


@router.post("/v1/loops/samplers")
async def create_sampler(
    body: dict, request: Request, service: TrainingService = Depends(training)
) -> dict:
    """Register a checkpoint-backed native Loops sampler."""
    if body.get("session_id") not in service.sessions:
        raise KeyError("Session not found")
    run_id = body.get("run_id")
    if run_id and run_id not in service.runs:
        raise KeyError("Run not found")
    if body.get("model_path"):
        service.checkpoint(body["model_path"])
    elif not run_id and body.get("base_model") not in service.models:
        raise ValueError("Unknown sampler base model")
    return {
        "sampler": {
            "id": run_id
            or uuid5(
                NAMESPACE_URL,
                f"{body['session_id']}:{body.get('model_path') or body.get('base_model')}",
            ).hex,
            "model_id": None,
            "base_url": str(request.base_url).rstrip("/"),
        }
    }


@router.get("/operations/{operation_id}")
async def retrieve(operation_id: str, service: TrainingService = Depends(training)) -> JSONResponse:
    """Poll a native Loops operation result."""
    operation = service.operations[operation_id]
    if not operation.done:
        return JSONResponse({"status": "pending"}, status_code=408)
    if operation.error:
        return JSONResponse(
            {"status": "error", "error": operation.error, "error_class": "ValueError"}
        )
    result = dict(operation.result or {})
    if operation.kind == "sample":
        result = as_completions(result, operation.payload)
    if isinstance(result.get("path"), str) and result["path"].startswith("tinker://"):
        result["path"] = (
            result["path"]
            .replace("tinker://", "bt://loops:", 1)
            .replace("/weights/", "/trainer_weights/", 1)
        )
    return JSONResponse({"status": "done", "result": result})


@router.post("/{operation}", status_code=202)
async def submit(operation: str, body: dict, service: TrainingService = Depends(training)) -> dict:
    """Submit a native Loops training operation."""
    mapping = {
        "init_trainer_server": "init",
        "forward": "forward",
        "forward_backward": "forward_backward",
        "optim_step": "optim_step",
        "save_state": "save_weights",
        "save_weights_for_sampler": "save_weights_for_sampler",
        "load_state": "load_weights",
        "load_state_with_optimizer": "load_weights",
    }
    if operation not in mapping:
        raise HTTPException(501, f"Unsupported local Loops operation: {operation}")
    run_id = body.get("run_id")
    if not run_id:
        raise HTTPException(422, "Use ServiceClient(base_url=...) so operations have a run_id")
    payload = {**body, "loops": True}
    if operation in {"forward", "forward_backward"}:
        if body.get("top_k_logprobs"):
            raise HTTPException(501, "Top-k forward scoring is not supported locally")
        payload["forward_backward_input"] = {
            key: body[key] for key in ("data", "loss_fn", "loss_fn_config") if key in body
        }
    if operation.startswith("load_state"):
        payload["optimizer"] = operation.endswith("with_optimizer")
    result = service.submit(run_id, body.get("seq_id"), mapping[operation], payload)
    return {"operation_id": result.id, "schema_version": 1}
