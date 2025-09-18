import logging
import uuid
from typing import Any, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel

from metta.app_backend.clients.base_client import NotAuthenticatedError, get_machine_token
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest, TaskFilterParams, TaskResponse, TasksResponse
from metta.app_backend.routes.score_routes import (
    PolicyScoresData,
    PolicyScoresRequest,
)
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

logger = logging.getLogger("stats_client")

T = TypeVar("T", bound=BaseModel)


class StatsClient:
    """Synchronous wrapper around AsyncStatsClient using httpx sync client."""

    def __init__(self, backend_url: str = PROD_STATS_SERVER_URI, machine_token: str | None = None):
        self._backend_url = backend_url
        self._http_client = httpx.Client(
            base_url=backend_url,
            timeout=30.0,
        )

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

    def _validate_authenticated(self) -> str:
        from metta.app_backend.server import WhoAmIResponse

        auth_user = self._make_sync_request(WhoAmIResponse, "GET", "/whoami")
        if auth_user.user_email in ["unknown", None]:
            raise NotAuthenticatedError(f"Not authenticated. User: {auth_user.user_email}")
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

    def update_training_run_status(self, run_id: uuid.UUID, status: str) -> None:
        headers = remove_none_values({"X-Auth-Token": self._machine_token})
        response = self._http_client.request(
            "PATCH", f"/stats/training-runs/{run_id}/status", headers=headers, json={"status": status}
        )
        response.raise_for_status()

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
        *,
        agent_policies: dict[int, uuid.UUID],
        agent_metrics: dict[int, dict[str, float]],
        primary_policy_id: uuid.UUID,
        sim_name: str,
        env_label: str,
        stats_epoch: uuid.UUID | None = None,
        replay_url: str | None = None,
        attributes: dict[str, Any] | None = None,
        eval_task_id: uuid.UUID | None = None,
        tags: list[str] | None = None,
        thumbnail_url: str | None = None,
    ) -> EpisodeResponse:
        data = EpisodeCreate(
            agent_policies=agent_policies,
            agent_metrics=agent_metrics,
            primary_policy_id=primary_policy_id,
            stats_epoch=stats_epoch,
            sim_name=sim_name,
            env_label=env_label,
            replay_url=replay_url,
            attributes=attributes or {},
            eval_task_id=eval_task_id,
            tags=tags,
            thumbnail_url=thumbnail_url,
        )
        return self._make_sync_request(EpisodeResponse, "POST", "/stats/episodes", json=data.model_dump(mode="json"))

    def create_task(self, request: TaskCreateRequest) -> TaskResponse:
        return self._make_sync_request(TaskResponse, "POST", "/tasks", json=request.model_dump(mode="json"))

    def get_all_tasks(self, filters: TaskFilterParams | None = None) -> TasksResponse:
        params = filters.model_dump(mode="json", exclude_none=True) if filters else {}
        return self._make_sync_request(TasksResponse, "GET", "/tasks/all", params=params)

    def get_policy_scores(self, request: PolicyScoresRequest) -> PolicyScoresData:
        return self._make_sync_request(
            PolicyScoresData, "POST", "/scorecard/score", json=request.model_dump(mode="json")
        )

    @staticmethod
    def create(stats_server_uri: str) -> Optional["StatsClient"]:
        machine_token = get_machine_token(stats_server_uri)
        if machine_token is None:
            raise NotAuthenticatedError(f"No machine token found for {stats_server_uri}")
        stats_client = StatsClient(backend_url=stats_server_uri, machine_token=machine_token)
        stats_client._validate_authenticated()
        return stats_client
