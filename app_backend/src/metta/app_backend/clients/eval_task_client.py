from typing import TypeVar

from pydantic import BaseModel

from metta.app_backend.clients.base_client import BaseAppBackendClient
from metta.app_backend.metta_repo import EvalTaskRow
from metta.app_backend.routes.eval_task_routes import (
    GitHashesRequest,
    GitHashesResponse,
    TaskAttemptsResponse,
    TaskClaimRequest,
    TaskClaimResponse,
    TaskCreateRequest,
    TaskFilterParams,
    TaskFinishRequest,
    TaskIdResponse,
    TasksResponse,
)

T = TypeVar("T", bound=BaseModel)


class EvalTaskClient(BaseAppBackendClient):
    def create_task(self, request: TaskCreateRequest) -> EvalTaskRow:
        return self._make_request(EvalTaskRow, "POST", "/tasks", json=request.model_dump(mode="json"))

    def get_available_tasks(self, limit: int = 200) -> TasksResponse:
        return self._make_request(TasksResponse, "GET", "/tasks/available", params={"limit": limit})

    def claim_tasks(self, request: TaskClaimRequest) -> TaskClaimResponse:
        return self._make_request(TaskClaimResponse, "POST", "/tasks/claim", json=request.model_dump(mode="json"))

    def get_task_by_id(self, task_id: str) -> EvalTaskRow:
        return self._make_request(EvalTaskRow, "GET", f"/tasks/{task_id}")

    def get_claimed_tasks(self, assignee: str | None = None) -> TasksResponse:
        params = {"assignee": assignee} if assignee is not None else {}
        return self._make_request(TasksResponse, "GET", "/tasks/claimed", params=params)

    def start_task(self, task_id: int) -> TaskIdResponse:
        return self._make_request(TaskIdResponse, "POST", f"/tasks/{task_id}/start")

    def finish_task(self, task_id: int, request: TaskFinishRequest) -> TaskIdResponse:
        return self._make_request(
            TaskIdResponse, "POST", f"/tasks/{task_id}/finish", json=request.model_dump(mode="json")
        )

    def get_git_hashes_for_workers(self, assignees: list[str]) -> GitHashesResponse:
        request = GitHashesRequest(assignees=assignees)
        return self._make_request(GitHashesResponse, "POST", "/tasks/git-hashes", json=request.model_dump(mode="json"))

    def get_latest_assigned_task_for_worker(self, assignee: str) -> EvalTaskRow | None:
        return self._make_request(EvalTaskRow, "GET", "/tasks/latest", params={"assignee": assignee})

    def get_all_tasks(self, filters: TaskFilterParams | None = None) -> TasksResponse:
        if filters is None:
            filters = TaskFilterParams()
        return self._make_request(
            TasksResponse, "GET", "/tasks/all", params=filters.model_dump(mode="json", exclude_none=True)
        )

    def get_task_attempts(self, task_id: int) -> TaskAttemptsResponse:
        return self._make_request(TaskAttemptsResponse, "GET", f"/tasks/{task_id}/attempts")
