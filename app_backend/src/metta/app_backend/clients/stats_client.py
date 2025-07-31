import uuid
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel

from metta.app_backend.clients.base_client import BaseAppBackendClient
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest, TaskFilterParams, TaskResponse, TasksResponse
from metta.app_backend.routes.stats_routes import (
    EpisodeCreate,
    EpisodeResponse,
    EpochCreate,
    EpochResponse,
    PolicyCreate,
    PolicyIdResponse,
    PolicyResponse,
    TrainingRunCreate,
    TrainingRunResponse,
)
from metta.common.util.collections import remove_none_values
from metta.common.util.constants import PROD_STATS_SERVER_URI

T = TypeVar("T", bound=BaseModel)


class AsyncStatsClient(BaseAppBackendClient):
    async def get_policy_ids(self, policy_names: list[str]) -> PolicyIdResponse:
        return await self._make_request(
            PolicyIdResponse, "GET", "/stats/policies/ids", params={"policy_names": policy_names}
        )

    async def create_training_run(
        self,
        name: str,
        attributes: dict[str, str] | None = None,
        url: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> TrainingRunResponse:
        data = TrainingRunCreate(
            name=name,
            attributes=attributes or {},
            url=url,
            description=description,
            tags=tags,
        )
        return await self._make_request(
            TrainingRunResponse, "POST", "/stats/training-runs", json=data.model_dump(mode="json")
        )

    async def create_epoch(
        self,
        run_id: uuid.UUID,
        start_training_epoch: int,
        end_training_epoch: int,
        attributes: dict[str, str] | None = None,
    ) -> EpochResponse:
        data = EpochCreate(
            start_training_epoch=start_training_epoch,
            end_training_epoch=end_training_epoch,
            attributes=attributes or {},
        )
        return await self._make_request(
            EpochResponse, "POST", f"/stats/training-runs/{run_id}/epochs", json=data.model_dump(mode="json")
        )

    async def create_policy(
        self,
        name: str,
        description: str | None = None,
        url: str | None = None,
        epoch_id: uuid.UUID | None = None,
    ) -> PolicyResponse:
        data = PolicyCreate(
            name=name,
            description=description,
            url=url,
            epoch_id=epoch_id,
        )
        return await self._make_request(PolicyResponse, "POST", "/stats/policies", json=data.model_dump(mode="json"))

    async def record_episode(
        self,
        agent_policies: dict[int, uuid.UUID],
        agent_metrics: dict[int, dict[str, float]],
        primary_policy_id: uuid.UUID,
        stats_epoch: uuid.UUID | None = None,
        eval_name: str | None = None,
        simulation_suite: str | None = None,
        replay_url: str | None = None,
        attributes: dict[str, Any] | None = None,
        eval_task_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
    ) -> EpisodeResponse:
        data = EpisodeCreate(
            agent_policies=agent_policies,
            agent_metrics=agent_metrics,
            primary_policy_id=primary_policy_id,
            stats_epoch=stats_epoch,
            eval_name=eval_name,
            simulation_suite=simulation_suite,
            replay_url=replay_url,
            attributes=attributes or {},
            eval_task_id=eval_task_id,
            tags=tags,
        )
        return await self._make_request(EpisodeResponse, "POST", "/stats/episodes", json=data.model_dump(mode="json"))


class StatsClient:
    """Synchronous wrapper around AsyncStatsClient using httpx sync client."""

    def __init__(self, backend_url: str = PROD_STATS_SERVER_URI, machine_token: str | None = None):
        self._backend_url = backend_url
        self._http_client = httpx.Client(
            base_url=backend_url,
            timeout=30.0,
        )

        from metta.common.util.stats_client_cfg import get_machine_token

        self.machine_token = machine_token or get_machine_token(backend_url)
        self._machine_token = self.machine_token

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()

    def close(self):
        self._http_client.close()

    def _make_sync_request(self, response_type: Type[T], method: str, url: str, **kwargs) -> T:
        headers = remove_none_values({"X-Auth-Token": self._machine_token})
        response = self._http_client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response_type.model_validate(response.json())

    def validate_authenticated(self) -> str:
        from metta.app_backend.server import WhoAmIResponse

        auth_user = self._make_sync_request(WhoAmIResponse, "GET", "/whoami")
        if auth_user.user_email in ["unknown", None]:
            raise ConnectionError(f"Not authenticated. User: {auth_user.user_email}")
        return auth_user.user_email

    def get_policy_ids(self, policy_names: list[str]) -> PolicyIdResponse:
        return self._make_sync_request(
            PolicyIdResponse, "GET", "/stats/policies/ids", params={"policy_names": policy_names}
        )

    def create_training_run(
        self,
        name: str,
        attributes: dict[str, str] | None = None,
        url: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> TrainingRunResponse:
        data = TrainingRunCreate(
            name=name,
            attributes=attributes or {},
            url=url,
            description=description,
            tags=tags,
        )
        return self._make_sync_request(
            TrainingRunResponse, "POST", "/stats/training-runs", json=data.model_dump(mode="json")
        )

    def create_epoch(
        self,
        run_id: uuid.UUID,
        start_training_epoch: int,
        end_training_epoch: int,
        attributes: dict[str, str] | None = None,
    ) -> EpochResponse:
        data = EpochCreate(
            start_training_epoch=start_training_epoch,
            end_training_epoch=end_training_epoch,
            attributes=attributes or {},
        )
        return self._make_sync_request(
            EpochResponse, "POST", f"/stats/training-runs/{run_id}/epochs", json=data.model_dump(mode="json")
        )

    def create_policy(
        self,
        name: str,
        description: str | None = None,
        url: str | None = None,
        epoch_id: uuid.UUID | None = None,
    ) -> PolicyResponse:
        data = PolicyCreate(
            name=name,
            description=description,
            url=url,
            epoch_id=epoch_id,
        )
        return self._make_sync_request(PolicyResponse, "POST", "/stats/policies", json=data.model_dump(mode="json"))

    def record_episode(
        self,
        agent_policies: dict[int, uuid.UUID],
        agent_metrics: dict[int, dict[str, float]],
        primary_policy_id: uuid.UUID,
        stats_epoch: uuid.UUID | None = None,
        eval_name: str | None = None,
        simulation_suite: str | None = None,
        replay_url: str | None = None,
        attributes: dict[str, Any] | None = None,
        eval_task_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
    ) -> EpisodeResponse:
        data = EpisodeCreate(
            agent_policies=agent_policies,
            agent_metrics=agent_metrics,
            primary_policy_id=primary_policy_id,
            stats_epoch=stats_epoch,
            eval_name=eval_name,
            simulation_suite=simulation_suite,
            replay_url=replay_url,
            attributes=attributes or {},
            eval_task_id=eval_task_id,
            tags=tags,
        )
        return self._make_sync_request(EpisodeResponse, "POST", "/stats/episodes", json=data.model_dump(mode="json"))

    def create_task(self, request: TaskCreateRequest) -> TaskResponse:
        return self._make_sync_request(TaskResponse, "POST", "/tasks", json=request.model_dump(mode="json"))

    def get_all_tasks(self, filters: TaskFilterParams | None = None) -> TasksResponse:
        params = filters.model_dump(mode="json", exclude_none=True) if filters else {}
        return self._make_sync_request(TasksResponse, "GET", "/tasks", params=params)
