import uuid
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from metta.app_backend.auth import create_user_or_token_dependency
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.metta_repo import TaskStatusUpdate as RepoTaskStatusUpdate
from metta.app_backend.route_logger import timed_route

TaskIdStr = str
TaskStatus = Literal["unprocessed", "canceled", "done", "error"]


class TaskCreateRequest(BaseModel):
    policy_id: str
    git_hash: str
    env_overrides: dict[str, Any] = Field(default_factory=dict)
    sim_suite: str = "all"


class TaskClaimRequest(BaseModel):
    eval_task_ids: list[TaskIdStr]
    assignee: str


class TaskStatusUpdate(BaseModel):
    status: TaskStatus
    details: dict[str, Any] | None = None


class TaskUpdateRequest(BaseModel):
    assignee: str
    statuses: dict[TaskIdStr, TaskStatusUpdate]


class TaskResponse(BaseModel):
    id: TaskIdStr
    policy_id: str
    sim_suite: str
    status: TaskStatus
    assigned_at: datetime | None = None
    assignee: str | None = None
    created_at: datetime
    attributes: dict[str, Any]


class AvailableTasksResponse(BaseModel):
    tasks: list[TaskResponse]


def create_eval_task_router(stats_repo: MettaRepo) -> APIRouter:
    router = APIRouter(prefix="/tasks", tags=["eval_tasks"])

    user_or_token = Depends(create_user_or_token_dependency(stats_repo))

    @router.post("", response_model=TaskResponse)
    @timed_route("create_task")
    async def create_task(request: TaskCreateRequest, user: str = user_or_token) -> TaskResponse:
        try:
            policy_uuid = uuid.UUID(request.policy_id)

            attributes = {
                "env_overrides": request.env_overrides,
                "git_hash": request.git_hash,
            }

            task_id = await stats_repo.create_eval_task(
                policy_id=policy_uuid,
                sim_suite=request.sim_suite,
                attributes=attributes,
            )

            return TaskResponse(
                id=str(task_id),
                policy_id=request.policy_id,
                sim_suite=request.sim_suite,
                status="unprocessed",
                assigned_at=None,
                assignee=None,
                created_at=datetime.utcnow(),
                attributes=attributes,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid UUID format") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}") from e

    @router.get("/available", response_model=AvailableTasksResponse)
    @timed_route("get_available_tasks")
    async def get_available_tasks(
        limit: int = Query(default=200, ge=1, le=1000), user: str = user_or_token
    ) -> AvailableTasksResponse:
        try:
            tasks = await stats_repo.get_available_tasks(limit=limit)

            task_responses = [
                TaskResponse(
                    id=str(task["id"]),
                    policy_id=str(task["policy_id"]),
                    sim_suite=task["sim_suite"],
                    status=task["status"],
                    assigned_at=task["assigned_at"],
                    assignee=task["assignee"],
                    created_at=task["created_at"],
                    attributes=task["attributes"] or {},
                )
                for task in tasks
            ]

            return AvailableTasksResponse(tasks=task_responses)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get available tasks: {str(e)}") from e

    @router.post("/claim")
    @timed_route("claim_tasks")
    async def claim_tasks(request: TaskClaimRequest, user: str = user_or_token) -> list[TaskIdStr]:
        try:
            task_uuids = [uuid.UUID(task_id) for task_id in request.eval_task_ids]

            claimed_ids = await stats_repo.claim_tasks(
                task_ids=task_uuids,
                assignee=request.assignee,
            )

            return [str(task_id) for task_id in claimed_ids]
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid UUID format") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to claim tasks: {str(e)}") from e

    @router.get("/claimed")
    @timed_route("get_claimed_tasks")
    async def get_claimed_tasks(assignee: str = Query(...), user: str = user_or_token) -> AvailableTasksResponse:
        try:
            tasks = await stats_repo.get_claimed_tasks(assignee=assignee)

            task_responses = [
                TaskResponse(
                    id=str(task["id"]),
                    policy_id=str(task["policy_id"]),
                    sim_suite=task["sim_suite"],
                    status=task["status"],
                    assigned_at=task["assigned_at"],
                    assignee=task["assignee"],
                    created_at=task["created_at"],
                    attributes=task["attributes"] or {},
                )
                for task in tasks
            ]

            return AvailableTasksResponse(tasks=task_responses)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get claimed tasks: {str(e)}") from e

    @router.post("/claimed/update")
    @timed_route("update_task_statuses")
    async def update_task_statuses(request: TaskUpdateRequest, user: str = user_or_token) -> dict[TaskIdStr, str]:
        try:
            task_updates = {
                uuid.UUID(task_id): RepoTaskStatusUpdate(status=status_update.status, details=status_update.details)
                for task_id, status_update in request.statuses.items()
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid format") from e

        try:
            updated = await stats_repo.update_task_statuses(
                assignee=request.assignee,
                task_updates=task_updates,
            )

            return {str(task_id): status for task_id, status in updated.items()}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update task statuses: {str(e)}") from e

    return router
