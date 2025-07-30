from typing import TypeVar

from pydantic import BaseModel

from metta.app_backend.clients.base_client import BaseAppBackendClient
from metta.app_backend.routes.eval_task_routes import (
    TaskClaimRequest,
    TaskClaimResponse,
    TaskCreateRequest,
    TaskFilterParams,
    TaskResponse,
    TasksResponse,
    TaskUpdateRequest,
    TaskUpdateResponse,
)
from metta.common.datadog.tracing import add_span_tags, trace_method

T = TypeVar("T", bound=BaseModel)


class EvalTaskClient(BaseAppBackendClient):
    async def create_task(self, request: TaskCreateRequest) -> TaskResponse:
        return await self._make_request(TaskResponse, "POST", "/tasks", json=request.model_dump(mode="json"))

    @trace_method(operation_name="eval.client.get_available_tasks", tags={"component": "client"})
    async def get_available_tasks(self, limit: int = 200) -> TasksResponse:
        add_span_tags({"limit": limit})
        return await self._make_request(TasksResponse, "GET", "/tasks/available", params={"limit": limit})

    @trace_method(operation_name="eval.client.claim_tasks", tags={"component": "client"})
    async def claim_tasks(self, request: TaskClaimRequest) -> TaskClaimResponse:
        add_span_tags({"num_tasks": len(request.tasks), "assignee": request.assignee})
        return await self._make_request(TaskClaimResponse, "POST", "/tasks/claim", json=request.model_dump(mode="json"))

    @trace_method(operation_name="eval.client.get_claimed_tasks", tags={"component": "client"})
    async def get_claimed_tasks(self, assignee: str | None = None) -> TasksResponse:
        if assignee:
            add_span_tags({"assignee": assignee})
        params = {"assignee": assignee} if assignee is not None else {}
        return await self._make_request(TasksResponse, "GET", "/tasks/claimed", params=params)

    @trace_method(operation_name="eval.client.update_task_status", tags={"component": "client"})
    async def update_task_status(self, request: TaskUpdateRequest) -> TaskUpdateResponse:
        add_span_tags({"num_updates": len(request.updates)})
        return await self._make_request(
            TaskUpdateResponse, "POST", "/tasks/claimed/update", json=request.model_dump(mode="json")
        )

    @trace_method(operation_name="eval.client.get_latest_task", tags={"component": "client"})
    async def get_latest_assigned_task_for_worker(self, assignee: str) -> TaskResponse:
        add_span_tags({"assignee": assignee})
        return await self._make_request(TaskResponse, "GET", "/tasks/latest", params={"assignee": assignee})

    @trace_method(operation_name="eval.client.get_all_tasks", tags={"component": "client"})
    async def get_all_tasks(self, filters: TaskFilterParams | None = None) -> TasksResponse:
        if filters is None:
            filters = TaskFilterParams()
        return await self._make_request(
            TasksResponse, "GET", "/tasks/all", params=filters.model_dump(mode="json", exclude_none=True)
        )
