from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from train.api.dependencies import training
from train.checkpoints import Checkpoint
from train.deployments import Deployment
from train.service import TrainingService

router = APIRouter(tags=["Baseten Loops deployment"])


class DeployCheckpoint(BaseModel):
    """Native Loops checkpoint IDs and deployment metadata."""

    checkpoint_ids: list[str] = Field(min_length=1)
    model_name: str = Field(min_length=1)
    instance_type_id: str
    hf_secret_name: str = Field(min_length=1)


@router.post("/v1/loops/checkpoints/deploy")
def deploy(body: DeployCheckpoint, service: TrainingService = Depends(training)) -> dict:
    """Deploy an immutable Loops sampler checkpoint."""
    records = {record.directory.name: record for record in _snapshots(service)}
    bindings = {}
    base = None
    for identifier in body.checkpoint_ids:
        record = records[identifier]
        uri, spec = service.deployments.pin(
            f"tinker://{'/'.join(record.directory.relative_to(service.root).parts)}"
        )
        name = record.uri.rsplit("/", 1)[-1]
        if name in bindings:
            raise ValueError("Checkpoint names in a deployment must be unique")
        if base is not None and base != spec.base_model:
            raise ValueError("All checkpoints must share a base model")
        bindings[name], base = uri, spec.base_model
    model_id, deployment_id = uuid4().hex, uuid4().hex
    service.deployments.create(
        f"baseten:{model_id}:{deployment_id}",
        bindings,
        base or "",
        metadata={
            "name": body.model_name,
            "instance_type_id": body.instance_type_id,
            "hf_secret_name": body.hf_secret_name,
        },
    )
    return {"model_id": model_id, "deployment_id": deployment_id}


@router.get("/v1/models/{model_id}/deployments/{deployment_id}")
def get(model_id: str, deployment_id: str, service: TrainingService = Depends(training)) -> dict:
    """Return a local Baseten deployment status."""
    deployment = service.deployments.registry().deployments[f"baseten:{model_id}:{deployment_id}"]
    return _info(deployment)


@router.get("/v1/models/{model_id}/deployments")
def listing(model_id: str, service: TrainingService = Depends(training)) -> dict:
    """List local Baseten model resources."""
    return {
        "deployments": [
            _info(deployment)
            for name, deployment in service.deployments.registry().deployments.items()
            if name.startswith(f"baseten:{model_id}:")
        ]
    }


@router.delete("/v1/models/{model_id}")
def delete(model_id: str, service: TrainingService = Depends(training)) -> dict:
    """Delete a local Baseten model and its deployment bindings."""
    deployments = [
        name
        for name in service.deployments.registry().deployments
        if name.startswith(f"baseten:{model_id}:")
    ]
    if not deployments:
        raise KeyError(model_id)
    for name in deployments:
        service.deployments.delete(name)
    return {"id": model_id, "deleted": True}


def _info(deployment: Deployment) -> dict:
    """Serialize a published adapter as a Baseten model resource."""
    _, model, identifier = deployment.id.split(":")
    return {
        "id": identifier,
        "model_id": model,
        "status": "ACTIVE",
        "created_at": deployment.created_at.isoformat(),
        "name": deployment.metadata.get("name", identifier),
        "is_production": False,
        "is_development": False,
        "active_replica_count": 1,
        "autoscaling_settings": None,
        "instance_type_name": "local",
        "environment": None,
        "request_backpressure_settings": {},
    }


def _snapshots(service: TrainingService) -> list[Checkpoint]:
    """List sampler snapshots as native Loops checkpoint resources."""
    return [
        Checkpoint.load(path)
        for path in sorted(service.root.glob("*/sampler_weights/*/checkpoint.json"))
    ]
