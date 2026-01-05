import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional, TypeVar
from urllib.parse import urlparse

import aioboto3
import boto3
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import gitta as git
from metta.app_backend.auth import UserOrToken
from metta.app_backend.metta_repo import EvalTaskRow, FinishedTaskStatus, MettaRepo, TaskAttemptRow, TaskStatus
from metta.app_backend.route_logger import timed_http_handler
from metta.common.util.git_repo import REPO_SLUG

OBSERVATORY_S3_BUCKET = "observatory-private"

T = TypeVar("T")


class TaskCreateRequest(BaseModel):
    command: str
    git_hash: str | None = None
    data_file: dict[str, Any] | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class TaskClaimRequest(BaseModel):
    tasks: list[int]
    assignee: str


class TaskClaimResponse(BaseModel):
    claimed: list[int]


class TaskFinishRequest(BaseModel):
    task_id: int
    status: FinishedTaskStatus
    log_path: str | None = None
    status_details: dict[str, Any] = Field(default_factory=dict)


class TaskIdResponse(BaseModel):
    task_id: int


class TaskFilterParams(BaseModel):
    limit: int = Field(default=500, ge=1, le=1000)
    statuses: list[str] | None = None
    git_hash: str | None = None


class TaskUpdateResponse(BaseModel):
    statuses: dict[int, TaskStatus]


class TasksResponse(BaseModel):
    tasks: list[EvalTaskRow]


class PaginatedTasksResponse(BaseModel):
    tasks: list[EvalTaskRow]
    total_count: int
    page: int
    page_size: int
    total_pages: int


class GitHashesRequest(BaseModel):
    assignees: list[str]


class GitHashesResponse(BaseModel):
    git_hashes: dict[str, list[str]]


class TaskAttemptsResponse(BaseModel):
    attempts: list[TaskAttemptRow]


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

    @router.post("")
    @timed_http_handler
    async def create_task(request: TaskCreateRequest, user: UserOrToken) -> EvalTaskRow:
        data_uri = None
        if request.data_file:
            file_path = f"data_file_{uuid.uuid4()}.json"
            with open(file_path, "w") as f:
                f.write(json.dumps(request.data_file))

            s3_key = f"eval_tasks/data_files/{uuid.uuid4()}.json"
            session = aioboto3.Session()
            async with session.client("s3") as s3_client:  # type: ignore
                await s3_client.upload_file(file_path, OBSERVATORY_S3_BUCKET, s3_key)
            data_uri = f"s3://{OBSERVATORY_S3_BUCKET}/{s3_key}"

            os.remove(file_path)

        task = await stats_repo.create_eval_task(
            command=request.command,
            git_hash=request.git_hash or await get_cached_latest_commit(),
            attributes=request.attributes,
            user_id=user,
            data_uri=data_uri,
        )
        return task

    @router.get("/latest", response_model=EvalTaskRow)
    @timed_http_handler
    async def get_latest_assigned_task_for_worker(assignee: str, _user: UserOrToken) -> EvalTaskRow | None:
        task = await stats_repo.get_latest_assigned_task_for_worker(assignee=assignee)
        return task

    @router.get("/available")
    @timed_http_handler
    async def get_available_tasks(
        _user: UserOrToken,
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> TasksResponse:
        tasks = await stats_repo.get_available_tasks(limit=limit)
        return TasksResponse(tasks=tasks)

    @router.post("/claim")
    @timed_http_handler
    async def claim_tasks(request: TaskClaimRequest, _user: UserOrToken) -> TaskClaimResponse:
        claimed_ids = await stats_repo.claim_tasks(
            task_ids=request.tasks,
            assignee=request.assignee,
        )
        return TaskClaimResponse(claimed=claimed_ids)

    @router.get("/claimed")
    @timed_http_handler
    async def get_claimed_tasks(_user: UserOrToken, assignee: str | None = Query(None)) -> TasksResponse:
        tasks = await stats_repo.get_claimed_tasks(assignee=assignee)
        return TasksResponse(tasks=tasks)

    @router.post("/git-hashes")
    @timed_http_handler
    async def get_git_hashes_for_workers(request: GitHashesRequest, _user: UserOrToken) -> GitHashesResponse:
        git_hashes = await stats_repo.get_git_hashes_for_workers(assignees=request.assignees)
        return GitHashesResponse(git_hashes=git_hashes)

    @router.get("/all")
    @timed_http_handler
    async def get_all_tasks(
        _user: UserOrToken,
        limit: int = Query(default=500, ge=1, le=1000),
        statuses: list[TaskStatus] | None = Query(default=None),
        git_hash: str | None = Query(default=None),
    ) -> TasksResponse:
        tasks = await stats_repo.get_all_tasks(
            limit=limit,
            statuses=statuses,
            git_hash=git_hash,
        )
        return TasksResponse(tasks=tasks)

    @router.get("/paginated")
    @timed_http_handler
    async def get_tasks_paginated(
        _user: UserOrToken,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=100),
        status: str | None = Query(default=None),
        assignee: str | None = Query(default=None),
        user_id: str | None = Query(default=None),
        command: str | None = Query(default=None),
        created_at: str | None = Query(default=None),
        assigned_at: str | None = Query(default=None),
    ) -> PaginatedTasksResponse:
        tasks, total_count = await stats_repo.get_tasks_paginated(
            page=page,
            page_size=page_size,
            status=status,
            assignee=assignee,
            user_id=user_id,
            command=command,
            created_at=created_at,
            assigned_at=assigned_at,
        )
        total_pages = (total_count + page_size - 1) // page_size
        return PaginatedTasksResponse(
            tasks=tasks,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    @router.post("/{task_id}/start")
    @timed_http_handler
    async def start_task(task_id: int, _user: UserOrToken) -> TaskIdResponse:
        await stats_repo.start_task(task_id=task_id)
        return TaskIdResponse(task_id=task_id)

    @router.post("/{task_id}/finish")
    @timed_http_handler
    async def finish_task(task_id: int, request: TaskFinishRequest, _user: UserOrToken) -> TaskIdResponse:
        await stats_repo.finish_task(
            task_id=task_id, status=request.status, status_details=request.status_details, log_path=request.log_path
        )
        return TaskIdResponse(task_id=task_id)

    @router.get("/{task_id}")
    @timed_http_handler
    async def get_task(task_id: int, _user: UserOrToken) -> EvalTaskRow:
        """Get a single task by ID with full details including attributes."""
        task = await stats_repo.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return task

    @router.get("/{task_id}/attempts")
    @timed_http_handler
    async def get_task_attempts(task_id: int, _user: UserOrToken) -> TaskAttemptsResponse:
        """Get all attempts for a specific task."""
        attempts = await stats_repo.get_task_attempts(task_id)
        return TaskAttemptsResponse(attempts=attempts)

    @router.get("/{task_id}/logs/{log_type}")
    @timed_http_handler
    async def get_task_logs(task_id: int, log_type: str, _user: UserOrToken):
        """Stream log files from S3 for a specific task.

        Args:
            task_id: The ID of the task
            log_type: Either "stdout" or "stderr" or "output"

        Returns:
            StreamingResponse with the log file content as text/plain
        """
        if log_type not in ("stdout", "stderr", "output"):
            raise HTTPException(status_code=400, detail="log_type must be 'stdout' or 'stderr' or 'output'")

        # Get the task to retrieve the log path from attributes
        task = await stats_repo.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # Get the log path from task attributes
        log_path = task.output_log_path

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
