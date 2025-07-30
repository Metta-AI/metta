import uuid
from datetime import datetime
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from metta.app_backend.auth import create_user_or_token_dependency
from metta.app_backend.metta_repo import MettaRepo, TaskStatus, TaskStatusUpdate
from metta.app_backend.route_logger import timed_http_handler
from metta.common.util.git import get_latest_commit

T = TypeVar("T")


class TaskCreateRequest(BaseModel):
    policy_id: uuid.UUID
    git_hash: str | None = None
    env_overrides: dict[str, Any] = Field(default_factory=dict)
    sim_suite: str = "all"


class TaskClaimRequest(BaseModel):
    tasks: list[uuid.UUID]
    assignee: str


class TaskClaimResponse(BaseModel):
    claimed: list[uuid.UUID]


class TaskUpdateRequest(BaseModel):
    require_assignee: str | None = None  # If supplied, the action only happens if the task is assigned to this worker
    updates: dict[uuid.UUID, TaskStatusUpdate]


class TaskFilterParams(BaseModel):
    limit: int = Field(default=500, ge=1, le=1000)
    statuses: list[str] | None = None
    git_hash: str | None = None
    policy_ids: list[uuid.UUID] | None = None
    sim_suites: list[str] | None = None


class TaskResponse(BaseModel):
    id: uuid.UUID
    policy_id: uuid.UUID
    sim_suite: str
    status: TaskStatus
    assigned_at: datetime | None = None
    assignee: str | None = None
    created_at: datetime
    attributes: dict[str, Any]
    policy_name: str | None = None
    retries: int

    def _attribute_property(self, key: str) -> Any | None:
        return self.attributes.get(key)

    @property
    def git_hash(self) -> str | None:
        return self._attribute_property("git_hash")

    @property
    def workers_spawned(self) -> int:
        return self._attribute_property("workers_spawned") or 0

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
            policy_name=task.get("policy_name"),
            retries=task["retries"],
        )


class TaskUpdateResponse(BaseModel):
    statuses: dict[uuid.UUID, TaskStatus]


class TasksResponse(BaseModel):
    tasks: list[TaskResponse]


def create_eval_task_router(stats_repo: MettaRepo) -> APIRouter:
    router = APIRouter(prefix="/tasks", tags=["eval_tasks"])

    user_or_token = Depends(create_user_or_token_dependency(stats_repo))

    @router.post("", response_model=TaskResponse)
    @timed_http_handler
    async def create_task(request: TaskCreateRequest, user: str = user_or_token) -> TaskResponse:
        # If no git_hash provided, fetch latest commit from main branch
        git_hash = request.git_hash
        if git_hash is None:
            git_hash = await get_latest_commit(branch="main")

        attributes = {
            "env_overrides": request.env_overrides,
            "git_hash": git_hash,
        }
        if not await stats_repo.get_policy_by_id(request.policy_id):
            raise HTTPException(status_code=404, detail=f"Policy {request.policy_id} not found")

        task = await stats_repo.create_eval_task(
            policy_id=request.policy_id,
            sim_suite=request.sim_suite,
            attributes=attributes,
        )
        return TaskResponse.from_db(task)

    @router.get("/latest", response_model=TaskResponse)
    @timed_http_handler
    async def get_latest_assigned_task_for_worker(assignee: str) -> TaskResponse | None:
        task = await stats_repo.get_latest_assigned_task_for_worker(assignee=assignee)
        return TaskResponse.from_db(task) if task else None

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

    @router.get("/all", response_model=TasksResponse)
    @timed_http_handler
    async def get_all_tasks(
        limit: int = Query(default=500, ge=1, le=1000),
        statuses: list[TaskStatus] | None = Query(default=None),
        git_hash: str | None = Query(default=None),
        policy_ids: list[uuid.UUID] | None = Query(default=None),
        sim_suites: list[str] | None = Query(default=None),
    ) -> TasksResponse:
        tasks = await stats_repo.get_all_tasks(
            limit=limit,
            statuses=statuses,
            git_hash=git_hash,
            policy_ids=policy_ids,
            sim_suites=sim_suites,
        )
        task_responses = [TaskResponse.from_db(task) for task in tasks]
        return TasksResponse(tasks=task_responses)

    @router.post("/claimed/update")
    @timed_http_handler
    async def update_task_statuses(request: TaskUpdateRequest) -> TaskUpdateResponse:
        updated = await stats_repo.update_task_statuses(
            updates=request.updates,
            require_assignee=request.require_assignee,
        )

        return TaskUpdateResponse(statuses=updated)

    return router
