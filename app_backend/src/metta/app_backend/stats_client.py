import uuid
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

from metta.app_backend.eval_task_client import EvalTaskClient
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest, TaskFilterParams, TaskResponse, TasksResponse
from metta.app_backend.routes.stats_routes import (
    EpisodeCreate,
    EpochCreate,
    PolicyCreate,
    PolicyIdResponse,
    TrainingRunCreate,
)


# Client-specific models with UUID fields
class ClientPolicyIdResponse(BaseModel):
    policy_ids: Dict[str, uuid.UUID]


class ClientTrainingRunResponse(BaseModel):
    id: uuid.UUID


class ClientEpochResponse(BaseModel):
    id: uuid.UUID


class ClientPolicyResponse(BaseModel):
    id: uuid.UUID


class ClientEpisodeResponse(BaseModel):
    id: uuid.UUID


class _NotAuthenticatedError(ConnectionError):
    """Exception raised when the stats client is not authenticated."""

    def __init__(self, message: str | None = None):
        super().__init__(
            message or "Unable to authenticate with the stats server. Run `metta status` to configure your token"
        )


class StatsClient:
    """Client for interacting with the stats API."""

    def __init__(self, http_client: httpx.Client, machine_token: str):
        """
        Initialize the stats client.

        Args:
            http_client: HTTP client implementation to use for requests
            machine_token: Machine token for authentication
        """
        self.http_client = http_client
        self.machine_token = machine_token

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()

    def close(self):
        """Close the HTTP client."""
        self.http_client.close()

    def validate_authenticated(self) -> str:
        auth_user = None
        try:
            response = self.http_client.get("/whoami", headers={"X-Auth-Token": self.machine_token})
            response.raise_for_status()
            if (auth_user := response.json().get("user_email")) not in ["unknown", None]:
                return auth_user
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise _NotAuthenticatedError(None) from e
            else:
                raise e
        raise _NotAuthenticatedError(auth_user and f"Authenticated as {auth_user}")

    def get_policy_ids(self, policy_names: list[str]) -> ClientPolicyIdResponse:
        """
        Get policy IDs for given policy names.

        Args:
            policy_names: List of policy names to get IDs for

        Returns:
            ClientPolicyIdResponse containing the mapping of names to UUIDs

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        params = {"policy_names": policy_names}
        headers = {"X-Auth-Token": self.machine_token}
        response = self.http_client.get("/stats/policies/ids", params=params, headers=headers)
        response.raise_for_status()

        # Deserialize string UUIDs to UUID objects
        server_response = PolicyIdResponse(**response.json())
        policy_ids_uuid = {name: uuid.UUID(uuid_str) for name, uuid_str in server_response.policy_ids.items()}
        return ClientPolicyIdResponse(policy_ids=policy_ids_uuid)

    def create_training_run(
        self,
        name: str,
        attributes: Optional[Dict[str, str]] = None,
        url: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ClientTrainingRunResponse:
        """
        Create a new training run.

        Args:
            name: Name of the training run
            attributes: Optional attributes for the training run
            url: Optional URL associated with the training run

        Returns:
            ClientTrainingRunResponse containing the created run UUID

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        data = TrainingRunCreate(name=name, attributes=attributes or {}, url=url, description=description, tags=tags)
        headers = {"X-Auth-Token": self.machine_token}
        response = self.http_client.post("/stats/training-runs", json=data.model_dump(), headers=headers)
        response.raise_for_status()

        # Deserialize string UUID to UUID object
        response_data = response.json()
        run_id_uuid = uuid.UUID(response_data["id"])
        return ClientTrainingRunResponse(id=run_id_uuid)

    def create_epoch(
        self,
        run_id: uuid.UUID,
        start_training_epoch: int,
        end_training_epoch: int,
        attributes: Optional[Dict[str, str]] = None,
    ) -> ClientEpochResponse:
        """
        Create a new policy epoch.

        Args:
            run_id: UUID of the training run
            start_training_epoch: Starting epoch number
            end_training_epoch: Ending epoch number
            attributes: Optional attributes for the epoch

        Returns:
            ClientPolicyEpochResponse containing the created epoch UUID

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        data = EpochCreate(
            start_training_epoch=start_training_epoch,
            end_training_epoch=end_training_epoch,
            attributes=attributes or {},
        )
        headers = {"X-Auth-Token": self.machine_token}
        response = self.http_client.post(
            f"/stats/training-runs/{run_id}/epochs", json=data.model_dump(), headers=headers
        )
        response.raise_for_status()

        # Deserialize string UUID to UUID object
        response_data = response.json()
        epoch_id_uuid = uuid.UUID(response_data["id"])
        return ClientEpochResponse(id=epoch_id_uuid)

    def create_policy(
        self,
        name: str,
        description: Optional[str] = None,
        url: Optional[str] = None,
        epoch_id: Optional[uuid.UUID] = None,
    ) -> ClientPolicyResponse:
        """
        Create a new policy.

        Args:
            name: Name of the policy
            description: Optional description of the policy
            url: Optional URL associated with the policy
            epoch_id: Optional UUID of the associated epoch

        Returns:
            ClientPolicyResponse containing the created policy UUID

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        epoch_id_str = str(epoch_id) if epoch_id else None
        data = PolicyCreate(name=name, description=description, url=url, epoch_id=epoch_id_str)
        headers = {"X-Auth-Token": self.machine_token}
        response = self.http_client.post("/stats/policies", json=data.model_dump(), headers=headers)
        response.raise_for_status()

        # Deserialize string UUID to UUID object
        response_data = response.json()
        policy_id_uuid = uuid.UUID(response_data["id"])
        return ClientPolicyResponse(id=policy_id_uuid)

    def record_episode(
        self,
        agent_policies: Dict[int, uuid.UUID],
        agent_metrics: Dict[int, Dict[str, float]],
        primary_policy_id: uuid.UUID,
        stats_epoch: Optional[uuid.UUID] = None,
        eval_name: Optional[str] = None,
        simulation_suite: Optional[str] = None,
        replay_url: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        eval_task_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
    ) -> ClientEpisodeResponse:
        """
        Record a new episode with agent policies and metrics.

        Args:
            agent_policies: Mapping of agent IDs to policy UUIDs
            agent_metrics: Mapping of agent IDs to their metrics
            primary_policy_id: UUID of the primary policy
            stats_epoch: Optional stats epoch UUID
            eval_name: Optional evaluation name
            simulation_suite: Optional simulation suite identifier
            replay_url: Optional URL to the replay
            attributes: Optional additional attributes
            eval_task_id: Optional UUID of the eval task this episode is for
            tags: Optional list of tags to associate with the episode

        Returns:
            ClientEpisodeResponse containing the created episode UUID

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        # Convert UUIDs to strings for serialization
        agent_policies_str = {agent_id: str(policy_id) for agent_id, policy_id in agent_policies.items()}
        primary_policy_id_str = str(primary_policy_id)
        stats_epoch_str = str(stats_epoch) if stats_epoch else None

        eval_task_id_str = str(eval_task_id) if eval_task_id else None
        data = EpisodeCreate(
            agent_policies=agent_policies_str,
            agent_metrics=agent_metrics,
            primary_policy_id=primary_policy_id_str,
            stats_epoch=stats_epoch_str,
            eval_name=eval_name,
            simulation_suite=simulation_suite,
            replay_url=replay_url,
            attributes=attributes or {},
            eval_task_id=eval_task_id_str,
            tags=tags,
        )
        headers = {"X-Auth-Token": self.machine_token}
        response = self.http_client.post("/stats/episodes", json=data.model_dump(), headers=headers)
        response.raise_for_status()

        # Deserialize string UUID to UUID object
        response_data = response.json()
        episode_id_uuid = uuid.UUID(response_data["id"])
        return ClientEpisodeResponse(id=episode_id_uuid)

    async def create_task(self, request: TaskCreateRequest) -> TaskResponse:
        client = EvalTaskClient(backend_url=str(self.http_client.base_url), machine_token=self.machine_token)
        return await client.create_task(request)

    async def get_all_tasks(self, filters: TaskFilterParams | None = None) -> TasksResponse:
        client = EvalTaskClient(backend_url=str(self.http_client.base_url), machine_token=self.machine_token)
        return await client.get_all_tasks(filters=filters)
