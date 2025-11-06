import typing

import pydantic

import metta.app_backend.clients.base_client
import metta.app_backend.routes.eval_task_routes

T = typing.TypeVar("T", bound=pydantic.BaseModel)


class EvalTaskClient(metta.app_backend.clients.base_client.BaseAppBackendClient):
    async def create_task(
        self, request: metta.app_backend.routes.eval_task_routes.TaskCreateRequest
    ) -> metta.app_backend.routes.eval_task_routes.TaskResponse:
        return await self._make_request(
            metta.app_backend.routes.eval_task_routes.TaskResponse,
            "POST",
            "/tasks",
            json=request.model_dump(mode="json"),
        )

    async def get_available_tasks(self, limit: int = 200) -> metta.app_backend.routes.eval_task_routes.TasksResponse:
        return await self._make_request(
            metta.app_backend.routes.eval_task_routes.TasksResponse, "GET", "/tasks/available", params={"limit": limit}
        )

    async def claim_tasks(
        self, request: metta.app_backend.routes.eval_task_routes.TaskClaimRequest
    ) -> metta.app_backend.routes.eval_task_routes.TaskClaimResponse:
        return await self._make_request(
            metta.app_backend.routes.eval_task_routes.TaskClaimResponse,
            "POST",
            "/tasks/claim",
            json=request.model_dump(mode="json"),
        )

    async def get_claimed_tasks(
        self, assignee: str | None = None
    ) -> metta.app_backend.routes.eval_task_routes.TasksResponse:
        params = {"assignee": assignee} if assignee is not None else {}
        return await self._make_request(
            metta.app_backend.routes.eval_task_routes.TasksResponse, "GET", "/tasks/claimed", params=params
        )

    async def update_task_status(
        self, request: metta.app_backend.routes.eval_task_routes.TaskUpdateRequest
    ) -> metta.app_backend.routes.eval_task_routes.TaskUpdateResponse:
        return await self._make_request(
            metta.app_backend.routes.eval_task_routes.TaskUpdateResponse,
            "POST",
            "/tasks/claimed/update",
            json=request.model_dump(mode="json"),
        )

    async def get_git_hashes_for_workers(
        self, assignees: list[str]
    ) -> metta.app_backend.routes.eval_task_routes.GitHashesResponse:
        request = metta.app_backend.routes.eval_task_routes.GitHashesRequest(assignees=assignees)
        return await self._make_request(
            metta.app_backend.routes.eval_task_routes.GitHashesResponse,
            "POST",
            "/tasks/git-hashes",
            json=request.model_dump(mode="json"),
        )

    async def get_latest_assigned_task_for_worker(
        self, assignee: str
    ) -> metta.app_backend.routes.eval_task_routes.TaskResponse | None:
        return await self._make_request(
            metta.app_backend.routes.eval_task_routes.TaskResponse,
            "GET",
            "/tasks/latest",
            params={"assignee": assignee},
        )

    async def get_all_tasks(
        self, filters: metta.app_backend.routes.eval_task_routes.TaskFilterParams | None = None
    ) -> metta.app_backend.routes.eval_task_routes.TasksResponse:
        if filters is None:
            filters = metta.app_backend.routes.eval_task_routes.TaskFilterParams()
        return await self._make_request(
            metta.app_backend.routes.eval_task_routes.TasksResponse,
            "GET",
            "/tasks/all",
            params=filters.model_dump(mode="json", exclude_none=True),
        )

    async def count_tasks(self, where_clause: str) -> metta.app_backend.routes.eval_task_routes.TaskCountResponse:
        return await self._make_request(
            metta.app_backend.routes.eval_task_routes.TaskCountResponse,
            "GET",
            "/tasks/count",
            params={"where_clause": where_clause},
        )

    async def get_avg_runtime(
        self, where_clause: str
    ) -> metta.app_backend.routes.eval_task_routes.TaskAvgRuntimeResponse:
        return await self._make_request(
            metta.app_backend.routes.eval_task_routes.TaskAvgRuntimeResponse,
            "GET",
            "/tasks/avg-runtime",
            params={"where_clause": where_clause},
        )
