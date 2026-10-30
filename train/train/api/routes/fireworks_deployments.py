import re
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from train.api.dependencies import training
from train.checkpoints import Checkpoint
from train.deployments import Deployment
from train.service import TrainingService

router = APIRouter(tags=["Fireworks deployment"])


@router.get("/v1/accounts")
def accounts() -> dict:
    """List the synthetic local Fireworks account."""
    return {"accounts": [{"name": "accounts/local"}]}


@router.get("/v1/accounts/{account}/rlorTrainerJobs/{job}/checkpoints")
def checkpoints(account: str, job: str, service: TrainingService = Depends(training)) -> dict:
    """List promotable sampler checkpoints from a local trainer run."""
    owner = job.removeprefix("run-")
    if not re.fullmatch(r"[0-9a-f]{32}", owner):
        raise ValueError("Expected a local trainer run ID")
    records = [
        Checkpoint.load(path) for path in sorted(service.root.glob(f"{owner}/*/*/checkpoint.json"))
    ]
    return {
        "checkpoints": [
            {
                "name": f"accounts/{account}/rlorTrainerJobs/{job}/checkpoints/{record.directory.name}",
                "createTime": record.metadata()["created_at"],
                "checkpointType": "CHECKPOINT_TYPE_INFERENCE_LORA"
                if record.directory.parent.name == "sampler_weights"
                else "CHECKPOINT_TYPE_TRAINING_LORA",
                "promotable": record.directory.parent.name == "sampler_weights",
            }
            for record in records
        ]
    }


@router.post("/v1/accounts/{account}/checkpoints/{checkpoint_id}:promote")
def promote(
    account: str, checkpoint_id: str, body: dict, service: TrainingService = Depends(training)
) -> dict:
    """Register an immutable checkpoint as a Fireworks model."""
    job = body["trainer_job_id"].rsplit("/", 1)[-1]
    name = body["output_model"]
    if not name.startswith(f"accounts/{account}/models/") or not re.fullmatch(
        r"[a-z0-9-]{1,63}", name.rsplit("/", 1)[-1]
    ):
        raise ValueError("Invalid output model resource name")
    service.deployments.promote(
        name, f"local/run-{job.removeprefix('run-')}/{checkpoint_id}", body["base_model"]
    )
    return {"model": _model_info(name, service)}


@router.get("/v1/accounts/{account}/models/{model_id}")
def model(account: str, model_id: str, service: TrainingService = Depends(training)) -> dict:
    """Return a promoted Fireworks model resource."""
    return _model_info(f"accounts/{account}/models/{model_id}", service)


@router.post("/v1/accounts/{account}/deployments")
def create(
    account: str,
    body: dict,
    deploymentId: str | None = None,
    service: TrainingService = Depends(training),
) -> dict:
    """Create a callable local Fireworks deployment."""
    manager = service.deployments
    if body.get("enableHotLoad") or body.get("forTraining"):
        raise HTTPException(
            501, "Cloud trainer provisioning and full-weight hotload are not supported locally"
        )
    model = body["baseModel"]
    bindings = {}
    if model in manager.registry().models:
        uri = manager.registry().models[model]
        _, spec = manager.pin(uri)
        bindings[model], base = uri, spec.base_model
    else:
        base = model
    deployment = manager.create(
        f"accounts/{account}/deployments/{deploymentId or uuid4().hex}",
        bindings,
        base,
        bool(body.get("enableAddons")),
    )
    return _deployment_info(deployment)


@router.get("/v1/accounts/{account}/deployments/{identifier}")
def get(account: str, identifier: str, service: TrainingService = Depends(training)) -> dict:
    """Return a local Fireworks deployment and readiness status."""
    return _deployment_info(
        service.deployments.registry().deployments[f"accounts/{account}/deployments/{identifier}"]
    )


@router.delete("/v1/accounts/{account}/deployments/{identifier}")
def delete(account: str, identifier: str, service: TrainingService = Depends(training)) -> dict:
    """Delete a local Fireworks deployment binding."""
    service.deployments.delete(f"accounts/{account}/deployments/{identifier}")
    return {}


@router.post("/v1/accounts/{account}/deployedModels")
def attach(account: str, body: dict, service: TrainingService = Depends(training)) -> dict:
    """Attach a promoted model to an addon-enabled deployment."""
    if not body["deployment"].startswith(f"accounts/{account}/deployments/"):
        raise ValueError("Deployment account does not match")
    service.deployments.attach(body["deployment"], body["model"])
    return {
        "name": f"accounts/{account}/deployedModels/{uuid4().hex}",
        "model": body["model"],
        "deployment": body["deployment"],
        "state": "DEPLOYED",
    }


def _model_info(name: str, service: TrainingService) -> dict:
    """Serialize a promoted checkpoint as a Fireworks model."""
    uri = service.deployments.registry().models[name]
    _, spec = service.deployments.pin(uri)
    return {
        "name": name,
        "state": "READY",
        "kind": "PEFT_ADDON",
        "peftDetails": {"baseModel": spec.base_model, "r": spec.rank},
    }


def _deployment_info(deployment: Deployment) -> dict:
    """Serialize a local deployment as a Fireworks resource."""
    return {
        "name": deployment.id,
        "baseModel": next(iter(deployment.models))
        if len(deployment.models) == 1 and not deployment.addons
        else deployment.base_model,
        "state": "READY",
        "enableAddons": deployment.addons,
    }
