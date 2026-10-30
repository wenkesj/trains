from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from train.api.routes import (
    completions,
    deployments,
    fireworks,
    fireworks_deployments,
    inference,
    loops,
    loops_deployments,
    tinker,
)
from train.runtime import Runtime
from train.service import TrainingService


def create_app(
    models: dict[str, Path] | None = None,
    root: Path = Path("var/train"),
    runtime: Runtime | None = None,
) -> FastAPI:
    """Create the local training API and mount native provider routes."""
    service = TrainingService(models or {}, root, runtime or Runtime())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        service.root.mkdir(parents=True, exist_ok=True)
        yield
        await service.close()

    app = FastAPI(title="Trains local training", lifespan=lifespan)
    app.state.training = service

    @app.exception_handler(KeyError)
    async def missing(request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse({"error": f"Resource not found: {exc}"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=422)

    _serving_errors(app)

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "models": list(service.models),
            "device": service.runtime.device,
            "loaded_trainers": sum(run.engine is not None for run in service.runs.values()),
        }

    for module in (
        deployments,
        fireworks_deployments,
        loops_deployments,
        inference,
        tinker,
        completions,
        fireworks,
        loops,
    ):
        app.include_router(module.router)
    return app


def _serving_errors(app: FastAPI) -> None:
    """Translate backend serving errors into their original HTTP response."""

    @app.exception_handler(httpx.HTTPStatusError)
    async def upstream_error(request: Request, exc: httpx.HTTPStatusError) -> JSONResponse:
        return JSONResponse(
            {
                "error": "Serving backend rejected the operation",
                "upstream_status": exc.response.status_code,
            },
            status_code=502,
        )

    @app.exception_handler(httpx.RequestError)
    async def unavailable(request: Request, exc: httpx.RequestError) -> JSONResponse:
        return JSONResponse({"error": "Serving backend unavailable"}, status_code=503)
