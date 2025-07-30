import asyncio
import uuid
from typing import Any

from metta.app_backend.clients.base_client import BaseAppBackendClient
from metta.app_backend.clients.eval_task_client import EvalTaskClient
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
    """Synchronous wrapper around AsyncStatsClient."""

    def __init__(self, backend_url: str, machine_token: str):
        self._async_client = AsyncStatsClient(backend_url=backend_url, machine_token=machine_token)
        self.machine_token = machine_token

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()

    def close(self):
        asyncio.run(self._async_client.close())

    def validate_authenticated(self) -> str:
        return asyncio.run(self._async_client.validate_authenticated())

    def get_policy_ids(self, policy_names: list[str]) -> PolicyIdResponse:
        return asyncio.run(self._async_client.get_policy_ids(policy_names))

    def create_training_run(
        self,
        name: str,
        attributes: dict[str, str] | None = None,
        url: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> TrainingRunResponse:
        return asyncio.run(self._async_client.create_training_run(name, attributes, url, description, tags))

    def create_epoch(
        self,
        run_id: uuid.UUID,
        start_training_epoch: int,
        end_training_epoch: int,
        attributes: dict[str, str] | None = None,
    ) -> EpochResponse:
        return asyncio.run(
            self._async_client.create_epoch(run_id, start_training_epoch, end_training_epoch, attributes)
        )

    def create_policy(
        self,
        name: str,
        description: str | None = None,
        url: str | None = None,
        epoch_id: uuid.UUID | None = None,
    ) -> PolicyResponse:
        return asyncio.run(self._async_client.create_policy(name, description, url, epoch_id))

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
        return asyncio.run(
            self._async_client.record_episode(
                agent_policies,
                agent_metrics,
                primary_policy_id,
                stats_epoch,
                eval_name,
                simulation_suite,
                replay_url,
                attributes,
                eval_task_id,
                tags,
            )
        )

    async def create_task(self, request: TaskCreateRequest) -> TaskResponse:
        async with EvalTaskClient.from_client(self._async_client) as client:
            return await client.create_task(request)

    async def get_all_tasks(self, filters: TaskFilterParams | None = None) -> TasksResponse:
        async with EvalTaskClient.from_client(self._async_client) as client:
            return await client.get_all_tasks(filters=filters)
