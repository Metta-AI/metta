from typing import Any, Dict, List, Optional

import httpx

from .stats_routes import (
    EpisodeCreate,
    EpisodeResponse,
    EpochCreate,
    PolicyCreate,
    PolicyEpochResponse,
    PolicyIdResponse,
    PolicyResponse,
    TrainingRunCreate,
    TrainingRunResponse,
)


class StatsClient:
    """Client for interacting with the stats API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        """
        Initialize the stats client.

        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def get_policy_ids(self, policy_names: List[str]) -> PolicyIdResponse:
        """
        Get policy IDs for given policy names.

        Args:
            policy_names: List of policy names to get IDs for

        Returns:
            PolicyIdResponse containing the mapping of names to IDs

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        params = {"policy_names": policy_names}
        response = await self._client.get(f"{self.base_url}/stats/policies/ids", params=params)
        response.raise_for_status()
        return PolicyIdResponse(**response.json())

    async def create_training_run(
        self, name: str, user_id: str, attributes: Optional[Dict[str, str]] = None, url: Optional[str] = None
    ) -> TrainingRunResponse:
        """
        Create a new training run.

        Args:
            name: Name of the training run
            user_id: ID of the user creating the run
            attributes: Optional attributes for the training run
            url: Optional URL associated with the training run

        Returns:
            TrainingRunResponse containing the created run ID

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        data = TrainingRunCreate(name=name, user_id=user_id, attributes=attributes or {}, url=url)
        response = await self._client.post(f"{self.base_url}/stats/training-runs", json=data.model_dump())
        response.raise_for_status()
        return TrainingRunResponse(**response.json())

    async def create_policy_epoch(
        self,
        run_id: int,
        start_training_epoch: int,
        end_training_epoch: int,
        attributes: Optional[Dict[str, str]] = None,
    ) -> PolicyEpochResponse:
        """
        Create a new policy epoch.

        Args:
            run_id: ID of the training run
            start_training_epoch: Starting epoch number
            end_training_epoch: Ending epoch number
            attributes: Optional attributes for the epoch

        Returns:
            PolicyEpochResponse containing the created epoch ID

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        data = EpochCreate(
            start_training_epoch=start_training_epoch,
            end_training_epoch=end_training_epoch,
            attributes=attributes or {},
        )
        response = await self._client.post(
            f"{self.base_url}/stats/training-runs/{run_id}/epochs", json=data.model_dump()
        )
        response.raise_for_status()
        return PolicyEpochResponse(**response.json())

    async def create_policy(
        self, name: str, description: Optional[str] = None, url: Optional[str] = None, epoch_id: Optional[int] = None
    ) -> PolicyResponse:
        """
        Create a new policy.

        Args:
            name: Name of the policy
            description: Optional description of the policy
            url: Optional URL associated with the policy
            epoch_id: Optional ID of the associated epoch

        Returns:
            PolicyResponse containing the created policy ID

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        data = PolicyCreate(name=name, description=description, url=url, epoch_id=epoch_id)
        response = await self._client.post(f"{self.base_url}/stats/policies", json=data.model_dump())
        response.raise_for_status()
        return PolicyResponse(**response.json())

    async def record_episode(
        self,
        agent_policies: Dict[int, int],
        agent_metrics: Dict[int, Dict[str, float]],
        primary_policy_id: int,
        training_epoch: Optional[int] = None,
        eval_name: Optional[str] = None,
        simulation_suite: Optional[str] = None,
        replay_url: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> EpisodeResponse:
        """
        Record a new episode with agent policies and metrics.

        Args:
            agent_policies: Mapping of agent IDs to policy IDs
            agent_metrics: Mapping of agent IDs to their metrics
            primary_policy_id: ID of the primary policy
            training_epoch: Optional training epoch number
            eval_name: Optional evaluation name
            simulation_suite: Optional simulation suite identifier
            replay_url: Optional URL to the replay
            attributes: Optional additional attributes

        Returns:
            EpisodeResponse containing the created episode ID

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        data = EpisodeCreate(
            agent_policies=agent_policies,
            agent_metrics=agent_metrics,
            primary_policy_id=primary_policy_id,
            training_epoch=training_epoch,
            eval_name=eval_name,
            simulation_suite=simulation_suite,
            replay_url=replay_url,
            attributes=attributes or {},
        )
        response = await self._client.post(f"{self.base_url}/stats/episodes", json=data.model_dump())
        response.raise_for_status()
        return EpisodeResponse(**response.json())
