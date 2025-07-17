import uuid
from datetime import datetime
from typing import Any, Literal, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from metta.app_backend.auth import create_user_or_token_dependency
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.metta_repo import TaskStatusUpdate as RepoTaskStatusUpdate
from metta.app_backend.route_logger import timed_http_handler

T = TypeVar("T")

TaskStatus = Literal["unprocessed", "canceled", "done", "error"]


class TaskCreateRequest(BaseModel):
    policy_id: uuid.UUID
    git_hash: str
    env_overrides: dict[str, Any] = Field(default_factory=dict)
    sim_suite: str = "all"


class TaskClaimRequest(BaseModel):
    tasks: list[uuid.UUID]
    assignee: str


class TaskClaimResponse(BaseModel):
    claimed: list[uuid.UUID]


class TaskStatusUpdate(BaseModel):
    status: TaskStatus
    details: dict[str, Any] | None = None


class TaskUpdateRequest(BaseModel):
    assignee: str
    statuses: dict[uuid.UUID, TaskStatusUpdate]


class TaskResponse(BaseModel):
    id: uuid.UUID
    policy_id: uuid.UUID
    sim_suite: str
    status: TaskStatus
    assigned_at: datetime | None = None
    assignee: str | None = None
    created_at: datetime
    attributes: dict[str, Any]

    @classmethod
    def from_db(cls, task: dict[str, Any]) -> "TaskResponse":
        return cls(
            id=task["id"],
            policy_id=task["policy_id"],
            sim_suite=task["sim_suite"],
            status=task["status"],
            assigned_at=task["assigned_at"],
            assignee=task["assignee"],
            created_at=task["created_at"],
            attributes=task["attributes"] or {},
        )


class TaskUpdateResponse(BaseModel):
    statuses: dict[uuid.UUID, str]


class TasksResponse(BaseModel):
    tasks: list[TaskResponse]


def create_eval_task_router(stats_repo: MettaRepo) -> APIRouter:
    router = APIRouter(prefix="/tasks", tags=["eval_tasks"])

    user_or_token = Depends(create_user_or_token_dependency(stats_repo))

    @router.post("", response_model=TaskResponse)
    @timed_http_handler
    async def create_task(request: TaskCreateRequest, user: str = user_or_token) -> TaskResponse:
        attributes = {
            "env_overrides": request.env_overrides,
            "git_hash": request.git_hash,
        }
        if not await stats_repo.get_policy_by_id(request.policy_id):
            raise HTTPException(status_code=404, detail=f"Policy {request.policy_id} not found")

        task = await stats_repo.create_eval_task(
            policy_id=request.policy_id,
            sim_suite=request.sim_suite,
            attributes=attributes,
        )
        return TaskResponse.from_db(task)

    @router.get("/available", response_model=TasksResponse)
    @timed_http_handler
    async def get_available_tasks(
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> TasksResponse:
        tasks = await stats_repo.get_available_tasks(limit=limit)
        task_responses = [TaskResponse.from_db(task) for task in tasks]
        return TasksResponse(tasks=task_responses)

    @router.post("/claim")
    @timed_http_handler
    async def claim_tasks(request: TaskClaimRequest) -> TaskClaimResponse:
        claimed_ids = await stats_repo.claim_tasks(
            task_ids=request.tasks,
            assignee=request.assignee,
        )
        return TaskClaimResponse(claimed=claimed_ids)

    @router.get("/claimed")
    @timed_http_handler
    async def get_claimed_tasks(assignee: str | None = Query(None)) -> TasksResponse:
        tasks = await stats_repo.get_claimed_tasks(assignee=assignee)
        task_responses = [TaskResponse.from_db(task) for task in tasks]
        return TasksResponse(tasks=task_responses)

    @router.get("/{task_id}", response_model=TaskResponse)
    @timed_http_handler
    async def get_task_by_id(task_id: uuid.UUID) -> TaskResponse:
        task = await stats_repo.get_task_by_id(task_id=task_id)

        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return TaskResponse.from_db(task)

    @router.post("/claimed/update")
    @timed_http_handler
    async def update_task_statuses(request: TaskUpdateRequest) -> TaskUpdateResponse:
        task_updates = {
            task_id: RepoTaskStatusUpdate(status=status_update.status, details=status_update.details)
            for task_id, status_update in request.statuses.items()
        }

        updated = await stats_repo.update_task_statuses(
            assignee=request.assignee,
            task_updates=task_updates,
        )

        return TaskUpdateResponse(statuses=updated)

    return router
