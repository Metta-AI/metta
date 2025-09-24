"""Sweep coordination routes for managing distributed training sweeps."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from metta.app_backend.auth import create_user_or_token_dependency
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.route_logger import timed_http_handler


# Request/Response Models
class SweepCreateRequest(BaseModel):
    project: str
    entity: str
    wandb_sweep_id: str


class SweepCreateResponse(BaseModel):
    created: bool
    sweep_id: uuid.UUID


class SweepInfo(BaseModel):
    exists: bool
    wandb_sweep_id: str


class RunIdResponse(BaseModel):
    run_id: str


def create_sweep_router(metta_repo: MettaRepo) -> APIRouter:
    """Create a sweep coordination router with the given MettaRepo instance."""
    router = APIRouter(prefix="/sweeps", tags=["sweeps"])

    # Create the user-or-token authentication dependency
    user_or_token = Depends(create_user_or_token_dependency(metta_repo))

    @router.post("/{sweep_name}/create_sweep", response_model=SweepCreateResponse)
    @timed_http_handler
    async def create_sweep(
        sweep_name: str, request: SweepCreateRequest, user: str = user_or_token
    ) -> SweepCreateResponse:
        """Initialize a new sweep or return existing sweep info (idempotent)."""
        # Check if sweep already exists
        existing_sweep = await metta_repo.get_sweep_by_name(sweep_name)

        if existing_sweep:
            return SweepCreateResponse(created=False, sweep_id=existing_sweep.id)

        # Create new sweep
        sweep_id = await metta_repo.create_sweep(
            name=sweep_name,
            project=request.project,
            entity=request.entity,
            wandb_sweep_id=request.wandb_sweep_id,
            user_id=user,
        )

        return SweepCreateResponse(created=True, sweep_id=sweep_id)

    @router.get("/{sweep_name}", response_model=SweepInfo)
    @timed_http_handler
    async def get_sweep(sweep_name: str, user: str = user_or_token) -> SweepInfo:
        """Get sweep information by name."""
        sweep = await metta_repo.get_sweep_by_name(sweep_name)

        if not sweep:
            return SweepInfo(exists=False, wandb_sweep_id="")

        return SweepInfo(exists=True, wandb_sweep_id=sweep.wandb_sweep_id)

    @router.post("/{sweep_name}/runs/next", response_model=RunIdResponse)
    @timed_http_handler
    async def get_next_run_id(sweep_name: str, user: str = user_or_token) -> RunIdResponse:
        """Get the next run ID for a sweep (atomic operation)."""
        sweep = await metta_repo.get_sweep_by_name(sweep_name)

        if not sweep:
            raise HTTPException(status_code=404, detail=f"Sweep '{sweep_name}' not found")

        # Atomically increment and get next run counter
        next_counter = await metta_repo.get_next_sweep_run_counter(sweep.id)

        # Format run ID as "sweep_name.r.counter"
        run_id = f"{sweep_name}.r.{next_counter}"

        return RunIdResponse(run_id=run_id)

    return router
