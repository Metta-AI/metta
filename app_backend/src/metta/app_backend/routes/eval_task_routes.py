import uuid
from datetime import datetime, timedelta
from typing import Any, Optional, TypeVar
from urllib.parse import urlparse

import boto3
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import gitta as git
from metta.app_backend.auth import create_user_or_token_dependency
from metta.app_backend.metta_repo import EvalTaskRow, EvalTaskWithPolicyName, MettaRepo, TaskStatus, TaskStatusUpdate
from metta.app_backend.route_logger import timed_http_handler
from metta.common.util.git_repo import REPO_SLUG

T = TypeVar("T")


class TaskCreateRequest(BaseModel):
    policy_id: uuid.UUID
    sim_suite: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    # We should remove these once clients have migrated
    git_hash: str | None = None
    env_overrides: dict[str, Any] = Field(default_factory=dict)


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


class TaskPaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=100)
    policy_name: str | None = None
    sim_suite: str | None = None
    status: str | None = None
    assignee: str | None = None
    user_id: str | None = None
    retries: str | None = None
    created_at: str | None = None
    assigned_at: str | None = None
    updated_at: str | None = None


class EvalTaskResponse(BaseModel):
    id: uuid.UUID
    policy_id: uuid.UUID
    policy_uri: str
    sim_suite: str
    status: TaskStatus
    assigned_at: datetime | None = None
    assignee: str | None = None
    created_at: datetime
    attributes: dict[str, Any]
    policy_name: str | None = None
    retries: int
    user_id: str | None = None
    updated_at: datetime

    def _attribute_property(self, key: str) -> Any | None:
        return self.attributes.get(key)

    @property
    def git_hash(self) -> str | None:
        return self._attribute_property("git_hash")

    @property
    def workers_spawned(self) -> int:
        return self._attribute_property("workers_spawned") or 0

    @classmethod
    def from_db(cls, row: EvalTaskWithPolicyName) -> "EvalTaskResponse":
        return cls(
            id=row.id,
            policy_id=row.policy_id,
            policy_uri=row.policy_url,
            sim_suite=row.sim_suite,
            status=row.status,  # type: ignore
            assigned_at=row.assigned_at,
            assignee=row.assignee,
            created_at=row.created_at,
            attributes=row.attributes,
            policy_name=row.policy_name,
            retries=row.retries,
            user_id=row.user_id,
            updated_at=row.updated_at,
        )


class TaskResponse(BaseModel):
    id: uuid.UUID
    policy_id: uuid.UUID
    sim_suite: str
    status: TaskStatus
    assigned_at: datetime | None = None
    assignee: str | None = None
    created_at: datetime
    attributes: dict[str, Any]
    retries: int
    user_id: str | None = None
    updated_at: datetime

    @classmethod
    def from_db(cls, task: EvalTaskRow) -> "TaskResponse":
        return cls(
            id=task.id,
            policy_id=task.policy_id,
            sim_suite=task.sim_suite,
            status=task.status,  # type: ignore
            assigned_at=task.assigned_at,
            assignee=task.assignee,
            created_at=task.created_at,
            attributes=task.attributes or {},
            retries=task.retries,
            user_id=task.user_id,
            updated_at=task.updated_at,
        )


class TaskUpdateResponse(BaseModel):
    statuses: dict[uuid.UUID, TaskStatus]


class TasksResponse(BaseModel):
    tasks: list[EvalTaskResponse]


class PaginatedTasksResponse(BaseModel):
    tasks: list[EvalTaskResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int


class GitHashesRequest(BaseModel):
    assignees: list[str]


class GitHashesResponse(BaseModel):
    git_hashes: dict[str, list[str]]


class TaskCountResponse(BaseModel):
    count: int


class TaskAvgRuntimeResponse(BaseModel):
    avg_runtime: float | None


def create_eval_task_router(stats_repo: MettaRepo) -> APIRouter:
    router = APIRouter(prefix="/tasks", tags=["eval_tasks"])

    # Cache for latest commit
    _latest_commit_cache: Optional[tuple[str, datetime]] = None
    _cache_ttl = timedelta(minutes=3)

    async def get_cached_latest_commit() -> str:
        nonlocal _latest_commit_cache

        now = datetime.now()
        if _latest_commit_cache:
            commit_hash, cached_time = _latest_commit_cache
            if now - cached_time < _cache_ttl:
                return commit_hash

        # Cache miss or expired - fetch new value
        commit_hash = await git.get_latest_commit(REPO_SLUG, branch="main")
        _latest_commit_cache = (commit_hash, now)
        return commit_hash

    user_or_token = Depends(create_user_or_token_dependency(stats_repo))

    @router.post("", response_model=TaskResponse)
    @timed_http_handler
    async def create_task(request: TaskCreateRequest, user: str = user_or_token) -> TaskResponse:
        # If no git_hash provided, fetch latest commit from main branch
        attributes = request.attributes.copy()
        if not attributes.get("git_hash"):
            if request.git_hash:
                # Remove this once clients have migrated
                attributes["git_hash"] = request.git_hash
            else:
                attributes["git_hash"] = await get_cached_latest_commit()

        policy = await stats_repo.get_policy_by_id(request.policy_id)
        if not policy:
            raise HTTPException(status_code=404, detail=f"Policy {request.policy_id} not found")

        if not policy.url:
            raise HTTPException(status_code=400, detail="Policy URL is not set")

        if not policy.url.startswith("s3://"):
            raise HTTPException(status_code=400, detail="Policy URL is not an S3 URL")

        task = await stats_repo.create_eval_task(
            policy_id=request.policy_id,
            sim_suite=request.sim_suite,
            attributes=attributes,
            user_id=user,
        )
        return TaskResponse.from_db(task)

    @router.get("/latest", response_model=EvalTaskResponse)
    @timed_http_handler
    async def get_latest_assigned_task_for_worker(assignee: str) -> EvalTaskResponse | None:
        task = await stats_repo.get_latest_assigned_task_for_worker(assignee=assignee)
        return EvalTaskResponse.from_db(task) if task else None

    @router.get("/available", response_model=TasksResponse)
    @timed_http_handler
    async def get_available_tasks(
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> TasksResponse:
        tasks = await stats_repo.get_available_tasks(limit=limit)
        task_responses = [EvalTaskResponse.from_db(task) for task in tasks]
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
        task_responses = [EvalTaskResponse.from_db(task) for task in tasks]
        return TasksResponse(tasks=task_responses)

    @router.post("/git-hashes")
    @timed_http_handler
    async def get_git_hashes_for_workers(request: GitHashesRequest) -> GitHashesResponse:
        git_hashes = await stats_repo.get_git_hashes_for_workers(assignees=request.assignees)
        return GitHashesResponse(git_hashes=git_hashes)

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
        task_responses = [EvalTaskResponse.from_db(task) for task in tasks]
        return TasksResponse(tasks=task_responses)

    @router.get("/paginated", response_model=PaginatedTasksResponse)
    @timed_http_handler
    async def get_tasks_paginated(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=100),
        policy_name: str | None = Query(default=None),
        sim_suite: str | None = Query(default=None),
        status: str | None = Query(default=None),
        assignee: str | None = Query(default=None),
        user_id: str | None = Query(default=None),
        retries: str | None = Query(default=None),
        created_at: str | None = Query(default=None),
        assigned_at: str | None = Query(default=None),
        updated_at: str | None = Query(default=None),
        include_attributes: bool = Query(default=False),
    ) -> PaginatedTasksResponse:
        tasks, total_count = await stats_repo.get_tasks_paginated(
            page=page,
            page_size=page_size,
            policy_name=policy_name,
            sim_suite=sim_suite,
            status=status,
            assignee=assignee,
            user_id=user_id,
            retries=retries,
            created_at=created_at,
            assigned_at=assigned_at,
            updated_at=updated_at,
            include_attributes=include_attributes,
        )
        task_responses = [EvalTaskResponse.from_db(task) for task in tasks]
        total_pages = (total_count + page_size - 1) // page_size
        return PaginatedTasksResponse(
            tasks=task_responses,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    @router.post("/claimed/update")
    @timed_http_handler
    async def update_task_statuses(request: TaskUpdateRequest) -> TaskUpdateResponse:
        updated = await stats_repo.update_task_statuses(
            updates=request.updates,
            require_assignee=request.require_assignee,
        )

        return TaskUpdateResponse(statuses=updated)

    @router.get("/count")
    @timed_http_handler
    async def count_tasks(where_clause: str = Query(default="")) -> TaskCountResponse:
        return TaskCountResponse(count=await stats_repo.count_tasks(where_clause=where_clause))

    @router.get("/avg-runtime")
    @timed_http_handler
    async def get_avg_runtime(where_clause: str = Query(default="")) -> TaskAvgRuntimeResponse:
        return TaskAvgRuntimeResponse(avg_runtime=await stats_repo.get_avg_runtime(where_clause=where_clause))

    @router.get("/{task_id}", response_model=EvalTaskResponse)
    @timed_http_handler
    async def get_task(task_id: uuid.UUID) -> EvalTaskResponse:
        """Get a single task by ID with full details including attributes."""
        task = await stats_repo.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return EvalTaskResponse.from_db(task)

    @router.get("/{task_id}/logs/{log_type}")
    @timed_http_handler
    async def get_task_logs(task_id: uuid.UUID, log_type: str):
        """Stream log files from S3 for a specific task.

        Args:
            task_id: The UUID of the task
            log_type: Either "stdout" or "stderr"

        Returns:
            StreamingResponse with the log file content as text/plain
        """
        if log_type not in ("stdout", "stderr"):
            raise HTTPException(status_code=400, detail="log_type must be 'stdout' or 'stderr'")

        # Get the task to retrieve the log path from attributes
        task = await stats_repo.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # Get the log path from task attributes
        log_path_key = f"{log_type}_log_path"
        log_path = task.attributes.get(log_path_key)

        if not log_path:
            raise HTTPException(status_code=404, detail=f"No {log_type} log path found for task {task_id}")

        # Parse the S3 URL (format: s3://bucket/key)
        parsed = urlparse(log_path)
        if parsed.scheme != "s3":
            raise HTTPException(status_code=400, detail=f"Invalid S3 URL format: {log_path}")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        # Stream the file from S3

        s3_client = boto3.client("s3")
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)

            # Return streaming response
            return StreamingResponse(
                response["Body"].iter_chunks(),
                media_type="text/plain",
                headers={"Content-Disposition": f'inline; filename="{task_id}_{log_type}.txt"'},
            )
        except s3_client.exceptions.NoSuchKey as e:
            raise HTTPException(status_code=404, detail=f"Log file not found in S3: {log_path}") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve log from S3: {str(e)}") from e

    return router
