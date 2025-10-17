from dataclasses import dataclass
from typing import ClassVar, Optional

import httpx

from metta.app_backend.clients.base_client import get_machine_token
from metta.app_backend.sweep_client import SweepClient


@dataclass(frozen=True)
class AgentBucketInfo:
    """Details about the S3 bucket used for Cogweb agent artifacts."""

    bucket: str
    prefix: str
    uri: str
    region: str

    @classmethod
    def from_dict(cls, payload: dict[str, str]) -> "AgentBucketInfo":
        return cls(
            bucket=payload["bucket"],
            prefix=payload.get("prefix", ""),
            uri=payload["uri"],
            region=payload.get("region", ""),
        )


class CogwebClient:
    """Unified client for interacting with all Cogweb backend services."""

    _instances: ClassVar[dict[tuple[str, str], "CogwebClient"]] = {}

    def __init__(self, base_url: str, auth_token: str):
        """Initialize the Cogweb client. Note: Use get_client() factory method for cached instances."""
        self._base_url = base_url
        self._auth_token = auth_token
        self._headers: dict[str, str] = {}
        if auth_token:
            self._headers["X-Auth-Token"] = auth_token
        self._sweep_client = SweepClient(base_url, auth_token)

    @classmethod
    def get_client(cls, base_url: str = "http://localhost:8000", auth_token: Optional[str] = None) -> "CogwebClient":
        """Factory method to get or create a cached CogwebClient instance."""
        # Resolve auth token if not provided
        if auth_token is None:
            auth_token = get_machine_token(base_url)

        # If still None, use empty string (SweepClient will handle authentication errors)
        if auth_token is None:
            auth_token = ""

        # Normalize base URL
        normalized_url = base_url.rstrip("/")

        # Check cache
        cache_key = (normalized_url, auth_token)
        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls(normalized_url, auth_token)

        return cls._instances[cache_key]

    @classmethod
    def clear_cache(cls):
        """Clear the instance cache. Useful for testing."""
        cls._instances.clear()

    def sweep_client(self) -> SweepClient:
        """Get the sweep client for direct access to sweep operations."""
        return self._sweep_client

    def get_agent_bucket(self) -> AgentBucketInfo:
        """Fetch the Cogweb agent bucket configuration."""

        response = httpx.get(f"{self._base_url}/agents/bucket", headers=self._headers)
        response.raise_for_status()
        return AgentBucketInfo.from_dict(response.json())

    # NOTE: Future service clients can be added here:
    # def stats_client(self) -> StatsClient:
    #     """Get the stats client for metrics and logging operations."""
    #     return self._stats_client
    #
    # def checkpoint_manager(self) -> CheckpointManager:
    #     """Get the checkpoint manager for model management."""
    #     return self._checkpoint_manager
