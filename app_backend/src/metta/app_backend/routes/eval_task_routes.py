import datetime
import typing
import urllib.parse
import uuid

import boto3
import fastapi
import fastapi.responses
import pydantic

import gitta as git
import metta.app_backend.auth
import metta.app_backend.metta_repo
import metta.app_backend.route_logger
import metta.common.util.git_repo

T = typing.TypeVar("T")


class TaskCreateRequest(pydantic.BaseModel):
    policy_id: uuid.UUID
    sim_suite: str
    attributes: dict[str, typing.Any] = pydantic.Field(default_factory=dict)

    # We should remove these once clients have migrated
    git_hash: str | None = None
    env_overrides: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


class TaskClaimRequest(pydantic.BaseModel):
    tasks: list[uuid.UUID]
    assignee: str


class TaskClaimResponse(pydantic.BaseModel):
    claimed: list[uuid.UUID]


class TaskUpdateRequest(pydantic.BaseModel):
    require_assignee: str | None = None  # If supplied, the action only happens if the task is assigned to this worker
    updates: dict[uuid.UUID, metta.app_backend.metta_repo.TaskStatusUpdate]


class TaskFilterParams(pydantic.BaseModel):
    limit: int = pydantic.Field(default=500, ge=1, le=1000)
    statuses: list[str] | None = None
    git_hash: str | None = None
    policy_ids: list[uuid.UUID] | None = None
    sim_suites: list[str] | None = None


class TaskPaginationParams(pydantic.BaseModel):
    page: int = pydantic.Field(default=1, ge=1)
    page_size: int = pydantic.Field(default=50, ge=1, le=100)
    policy_name: str | None = None
    sim_suite: str | None = None
    status: str | None = None
    assignee: str | None = None
    user_id: str | None = None
    retries: str | None = None
    created_at: str | None = None
    assigned_at: str | None = None
    updated_at: str | None = None


class EvalTaskResponse(pydantic.BaseModel):
    id: uuid.UUID
    policy_id: uuid.UUID
    policy_uri: str
    sim_suite: str
    status: metta.app_backend.metta_repo.TaskStatus
    assigned_at: datetime.datetime | None = None
    assignee: str | None = None
    created_at: datetime.datetime
    attributes: dict[str, typing.Any]
    policy_name: str | None = None
    retries: int
    user_id: str | None = None
    updated_at: datetime.datetime

    def _attribute_property(self, key: str) -> typing.Any | None:
        return self.attributes.get(key)

    @property
    def git_hash(self) -> str | None:
        return self._attribute_property("git_hash")

    @property
    def workers_spawned(self) -> int:
        return self._attribute_property("workers_spawned") or 0

    @classmethod
    def from_db(cls, row: metta.app_backend.metta_repo.EvalTaskWithPolicyName) -> "EvalTaskResponse":
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


class TaskResponse(pydantic.BaseModel):
    id: uuid.UUID
    policy_id: uuid.UUID
    sim_suite: str
    status: metta.app_backend.metta_repo.TaskStatus
    assigned_at: datetime.datetime | None = None
    assignee: str | None = None
    created_at: datetime.datetime
    attributes: dict[str, typing.Any]
    retries: int
    user_id: str | None = None
    updated_at: datetime.datetime

    @classmethod
    def from_db(cls, task: metta.app_backend.metta_repo.EvalTaskRow) -> "TaskResponse":
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


class TaskUpdateResponse(pydantic.BaseModel):
    statuses: dict[uuid.UUID, metta.app_backend.metta_repo.TaskStatus]


class TasksResponse(pydantic.BaseModel):
    tasks: list[EvalTaskResponse]


class PaginatedTasksResponse(pydantic.BaseModel):
    tasks: list[EvalTaskResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int


class GitHashesRequest(pydantic.BaseModel):
    assignees: list[str]


class GitHashesResponse(pydantic.BaseModel):
    git_hashes: dict[str, list[str]]


class TaskCountResponse(pydantic.BaseModel):
    count: int


class TaskAvgRuntimeResponse(pydantic.BaseModel):
    avg_runtime: float | None


def create_eval_task_router(stats_repo: metta.app_backend.metta_repo.MettaRepo) -> fastapi.APIRouter:
    router = fastapi.APIRouter(prefix="/tasks", tags=["eval_tasks"])

    # Cache for latest commit
    _latest_commit_cache: typing.Optional[tuple[str, datetime.datetime]] = None
    _cache_ttl = datetime.timedelta(minutes=3)

    async def get_cached_latest_commit() -> str:
        nonlocal _latest_commit_cache

        now = datetime.datetime.now()
        if _latest_commit_cache:
            commit_hash, cached_time = _latest_commit_cache
            if now - cached_time < _cache_ttl:
                return commit_hash

        # Cache miss or expired - fetch new value
        commit_hash = await git.get_latest_commit(metta.common.util.git_repo.REPO_SLUG, branch="main")
        _latest_commit_cache = (commit_hash, now)
        return commit_hash

    user_or_token = fastapi.Depends(metta.app_backend.auth.create_user_or_token_dependency(stats_repo))

    @router.post("", response_model=TaskResponse)
    @metta.app_backend.route_logger.timed_http_handler
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
            raise fastapi.HTTPException(status_code=404, detail=f"Policy {request.policy_id} not found")

        if not policy.url:
            raise fastapi.HTTPException(status_code=400, detail="Policy URL is not set")

        if not policy.url.startswith("s3://"):
            raise fastapi.HTTPException(status_code=400, detail="Policy URL is not an S3 URL")

        task = await stats_repo.create_eval_task(
            policy_id=request.policy_id,
            sim_suite=request.sim_suite,
            attributes=attributes,
            user_id=user,
        )
        return TaskResponse.from_db(task)

    @router.get("/latest", response_model=EvalTaskResponse)
    @metta.app_backend.route_logger.timed_http_handler
    async def get_latest_assigned_task_for_worker(assignee: str) -> EvalTaskResponse | None:
        task = await stats_repo.get_latest_assigned_task_for_worker(assignee=assignee)
        return EvalTaskResponse.from_db(task) if task else None

    @router.get("/available", response_model=TasksResponse)
    @metta.app_backend.route_logger.timed_http_handler
    async def get_available_tasks(
        limit: int = fastapi.Query(default=200, ge=1, le=1000),
    ) -> TasksResponse:
        tasks = await stats_repo.get_available_tasks(limit=limit)
        task_responses = [EvalTaskResponse.from_db(task) for task in tasks]
        return TasksResponse(tasks=task_responses)

    @router.post("/claim")
    @metta.app_backend.route_logger.timed_http_handler
    async def claim_tasks(request: TaskClaimRequest) -> TaskClaimResponse:
        claimed_ids = await stats_repo.claim_tasks(
            task_ids=request.tasks,
            assignee=request.assignee,
        )
        return TaskClaimResponse(claimed=claimed_ids)

    @router.get("/claimed")
    @metta.app_backend.route_logger.timed_http_handler
    async def get_claimed_tasks(assignee: str | None = fastapi.Query(None)) -> TasksResponse:
        tasks = await stats_repo.get_claimed_tasks(assignee=assignee)
        task_responses = [EvalTaskResponse.from_db(task) for task in tasks]
        return TasksResponse(tasks=task_responses)

    @router.post("/git-hashes")
    @metta.app_backend.route_logger.timed_http_handler
    async def get_git_hashes_for_workers(request: GitHashesRequest) -> GitHashesResponse:
        git_hashes = await stats_repo.get_git_hashes_for_workers(assignees=request.assignees)
        return GitHashesResponse(git_hashes=git_hashes)

    @router.get("/all", response_model=TasksResponse)
    @metta.app_backend.route_logger.timed_http_handler
    async def get_all_tasks(
        limit: int = fastapi.Query(default=500, ge=1, le=1000),
        statuses: list[metta.app_backend.metta_repo.TaskStatus] | None = fastapi.Query(default=None),
        git_hash: str | None = fastapi.Query(default=None),
        policy_ids: list[uuid.UUID] | None = fastapi.Query(default=None),
        sim_suites: list[str] | None = fastapi.Query(default=None),
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
    @metta.app_backend.route_logger.timed_http_handler
    async def get_tasks_paginated(
        page: int = fastapi.Query(default=1, ge=1),
        page_size: int = fastapi.Query(default=50, ge=1, le=100),
        policy_name: str | None = fastapi.Query(default=None),
        sim_suite: str | None = fastapi.Query(default=None),
        status: str | None = fastapi.Query(default=None),
        assignee: str | None = fastapi.Query(default=None),
        user_id: str | None = fastapi.Query(default=None),
        retries: str | None = fastapi.Query(default=None),
        created_at: str | None = fastapi.Query(default=None),
        assigned_at: str | None = fastapi.Query(default=None),
        updated_at: str | None = fastapi.Query(default=None),
        include_attributes: bool = fastapi.Query(default=False),
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
    @metta.app_backend.route_logger.timed_http_handler
    async def update_task_statuses(request: TaskUpdateRequest) -> TaskUpdateResponse:
        updated = await stats_repo.update_task_statuses(
            updates=request.updates,
            require_assignee=request.require_assignee,
        )

        return TaskUpdateResponse(statuses=updated)

    @router.get("/count")
    @metta.app_backend.route_logger.timed_http_handler
    async def count_tasks(where_clause: str = fastapi.Query(default="")) -> TaskCountResponse:
        return TaskCountResponse(count=await stats_repo.count_tasks(where_clause=where_clause))

    @router.get("/avg-runtime")
    @metta.app_backend.route_logger.timed_http_handler
    async def get_avg_runtime(where_clause: str = fastapi.Query(default="")) -> TaskAvgRuntimeResponse:
        return TaskAvgRuntimeResponse(avg_runtime=await stats_repo.get_avg_runtime(where_clause=where_clause))

    @router.get("/{task_id}", response_model=EvalTaskResponse)
    @metta.app_backend.route_logger.timed_http_handler
    async def get_task(task_id: uuid.UUID) -> EvalTaskResponse:
        """Get a single task by ID with full details including attributes."""
        task = await stats_repo.get_task_by_id(task_id)
        if not task:
            raise fastapi.HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return EvalTaskResponse.from_db(task)

    @router.get("/{task_id}/logs/{log_type}")
    @metta.app_backend.route_logger.timed_http_handler
    async def get_task_logs(task_id: uuid.UUID, log_type: str):
        """Stream log files from S3 for a specific task.

        Args:
            task_id: The UUID of the task
            log_type: Either "stdout" or "stderr" or "output"

        Returns:
            StreamingResponse with the log file content as text/plain
        """
        if log_type not in ("stdout", "stderr", "output"):
            raise fastapi.HTTPException(status_code=400, detail="log_type must be 'stdout' or 'stderr' or 'output'")

        # Get the task to retrieve the log path from attributes
        task = await stats_repo.get_task_by_id(task_id)
        if not task:
            raise fastapi.HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # Get the log path from task attributes
        log_path_key = f"{log_type}_log_path"
        log_path = task.attributes.get(log_path_key)

        if not log_path:
            raise fastapi.HTTPException(status_code=404, detail=f"No {log_type} log path found for task {task_id}")

        # Parse the S3 URL (format: s3://bucket/key)
        parsed = urllib.parse.urlparse(log_path)
        if parsed.scheme != "s3":
            raise fastapi.HTTPException(status_code=400, detail=f"Invalid S3 URL format: {log_path}")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        # Stream the file from S3

        s3_client = boto3.client("s3")
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)

            # Return streaming response
            return fastapi.responses.StreamingResponse(
                response["Body"].iter_chunks(),
                media_type="text/plain",
                headers={"Content-Disposition": f'inline; filename="{task_id}_{log_type}.txt"'},
            )
        except s3_client.exceptions.NoSuchKey as e:
            raise fastapi.HTTPException(status_code=404, detail=f"Log file not found in S3: {log_path}") from e
        except Exception as e:
            raise fastapi.HTTPException(status_code=500, detail=f"Failed to retrieve log from S3: {str(e)}") from e

    return router
