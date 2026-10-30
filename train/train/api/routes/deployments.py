from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import Field

from train.api.dependencies import training
from train.deployments import Deployment
from train.models import Model
from train.service import TrainingService

router = APIRouter(tags=["Local deployments"])


class Publish(Model):
    """A checkpoint-to-model publication request."""

    checkpoint: str
    model: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")


class Publication(Model):
    """The callable endpoint and status of a published model."""

    id: str
    model: str
    checkpoint: str
    base_url: str
    protocol: Literal["openai-chat"] = "openai-chat"
    status: Literal["ready"] = "ready"


@router.post("/v1/deployments")
def publish(
    body: Publish, request: Request, service: TrainingService = Depends(training)
) -> Publication:
    """Publish a sampler checkpoint and return its callable endpoint."""
    manager = service.deployments
    uri, spec = manager.pin(body.checkpoint)
    deployment = manager.create(body.model, {body.model: uri}, spec.base_model)
    return _receipt(deployment, request)


@router.get("/v1/deployments/{identifier}")
def get(
    identifier: str, request: Request, service: TrainingService = Depends(training)
) -> Publication:
    """Return the current publication receipt for a model."""
    deployment = service.deployments.registry().deployments[identifier]
    return _receipt(deployment, request)


@router.delete("/v1/deployments/{identifier}")
def delete(identifier: str, service: TrainingService = Depends(training)) -> dict:
    """Remove a publication binding and unload unused adapters."""
    service.deployments.delete(identifier)
    return {"id": identifier, "deleted": True}


def _receipt(deployment: Deployment, request: Request) -> Publication:
    """Serialize a deployment binding as an API publication receipt."""
    return Publication(
        id=deployment.id,
        model=deployment.id,
        checkpoint=next(iter(deployment.models.values())),
        base_url=str(request.base_url).rstrip("/") + "/v1",
    )
