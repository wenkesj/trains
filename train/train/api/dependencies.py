from fastapi import Request

from train.service import TrainingService


def training(request: Request) -> TrainingService:
    """Resolve the training service from the active application."""
    return request.app.state.training
