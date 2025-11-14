from typing import TypeVar

from pydantic import BaseModel

from metta.app_backend.clients.base_client import BaseAppBackendClient
from metta.app_backend.metta_repo import EvalTaskRow
from metta.app_backend.routes.eval_task_routes import (
    EvalTaskResponse,
    GitHashesRequest,
    GitHashesResponse,
    TaskAvgRuntimeResponse,
    TaskClaimRequest,
    TaskClaimResponse,
    TaskCountResponse,
    TaskCreateRequest,
    TaskFilterParams,
    TaskFinishRequest,
    TaskIdResponse,
    TasksResponse,
)

T = TypeVar("T", bound=BaseModel)


class EvalTaskClient(BaseAppBackendClient):
    async def create_task(self, request: TaskCreateRequest) -> EvalTaskRow:
        return await self._make_request(EvalTaskRow, "POST", "/tasks", json=request.model_dump(mode="json"))

    async def get_available_tasks(self, limit: int = 200) -> TasksResponse:
        return await self._make_request(TasksResponse, "GET", "/tasks/available", params={"limit": limit})

    async def claim_tasks(self, request: TaskClaimRequest) -> TaskClaimResponse:
        return await self._make_request(TaskClaimResponse, "POST", "/tasks/claim", json=request.model_dump(mode="json"))

    async def get_task_by_id(self, task_id: str) -> EvalTaskResponse:
        return await self._make_request(EvalTaskResponse, "GET", f"/tasks/{task_id}")

    async def get_claimed_tasks(self, assignee: str | None = None) -> TasksResponse:
        params = {"assignee": assignee} if assignee is not None else {}
        return await self._make_request(TasksResponse, "GET", "/tasks/claimed", params=params)

    async def start_task(self, task_id: int) -> TaskIdResponse:
        return await self._make_request(TaskIdResponse, "POST", f"/tasks/{task_id}/start")

    async def finish_task(self, task_id: int, request: TaskFinishRequest) -> TaskIdResponse:
        return await self._make_request(
            TaskIdResponse, "POST", f"/tasks/{task_id}/finish", json=request.model_dump(mode="json")
        )

    async def get_git_hashes_for_workers(self, assignees: list[str]) -> GitHashesResponse:
        request = GitHashesRequest(assignees=assignees)
        return await self._make_request(
            GitHashesResponse, "POST", "/tasks/git-hashes", json=request.model_dump(mode="json")
        )

    async def get_latest_assigned_task_for_worker(self, assignee: str) -> EvalTaskRow | None:
        return await self._make_request(EvalTaskRow, "GET", "/tasks/latest", params={"assignee": assignee})

    async def get_all_tasks(self, filters: TaskFilterParams | None = None) -> TasksResponse:
        if filters is None:
            filters = TaskFilterParams()
        return await self._make_request(
            TasksResponse, "GET", "/tasks/all", params=filters.model_dump(mode="json", exclude_none=True)
        )

    async def count_tasks(self, where_clause: str) -> TaskCountResponse:
        return await self._make_request(TaskCountResponse, "GET", "/tasks/count", params={"where_clause": where_clause})

    async def get_avg_runtime(self, where_clause: str) -> TaskAvgRuntimeResponse:
        return await self._make_request(
            TaskAvgRuntimeResponse, "GET", "/tasks/avg-runtime", params={"where_clause": where_clause}
        )
