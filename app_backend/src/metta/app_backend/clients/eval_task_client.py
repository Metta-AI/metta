from typing import TypeVar

from pydantic import BaseModel

from metta.app_backend.clients.base_client import BaseAppBackendClient
from metta.app_backend.routes.eval_task_routes import (
    GitHashesRequest,
    GitHashesResponse,
    TaskClaimRequest,
    TaskClaimResponse,
    TaskCreateRequest,
    TaskFilterParams,
    TaskResponse,
    TasksResponse,
    TaskUpdateRequest,
    TaskUpdateResponse,
)

T = TypeVar("T", bound=BaseModel)


class EvalTaskClient(BaseAppBackendClient):
    async def create_task(self, request: TaskCreateRequest) -> TaskResponse:
        return await self._make_request(TaskResponse, "POST", "/tasks", json=request.model_dump(mode="json"))

    async def get_available_tasks(self, limit: int = 200) -> TasksResponse:
        return await self._make_request(TasksResponse, "GET", "/tasks/available", params={"limit": limit})

    async def claim_tasks(self, request: TaskClaimRequest) -> TaskClaimResponse:
        return await self._make_request(TaskClaimResponse, "POST", "/tasks/claim", json=request.model_dump(mode="json"))

    async def get_claimed_tasks(self, assignee: str | None = None) -> TasksResponse:
        params = {"assignee": assignee} if assignee is not None else {}
        return await self._make_request(TasksResponse, "GET", "/tasks/claimed", params=params)

    async def update_task_status(self, request: TaskUpdateRequest) -> TaskUpdateResponse:
        return await self._make_request(
            TaskUpdateResponse, "POST", "/tasks/claimed/update", json=request.model_dump(mode="json")
        )

    async def get_git_hashes_for_workers(self, assignees: list[str]) -> GitHashesResponse:
        request = GitHashesRequest(assignees=assignees)
        return await self._make_request(
            GitHashesResponse, "POST", "/tasks/git-hashes", json=request.model_dump(mode="json")
        )

    async def get_latest_assigned_task_for_worker(self, assignee: str) -> TaskResponse:
        return await self._make_request(TaskResponse, "GET", "/tasks/latest", params={"assignee": assignee})

    async def get_all_tasks(self, filters: TaskFilterParams | None = None) -> TasksResponse:
        if filters is None:
            filters = TaskFilterParams()
        return await self._make_request(
            TasksResponse, "GET", "/tasks/all", params=filters.model_dump(mode="json", exclude_none=True)
        )
