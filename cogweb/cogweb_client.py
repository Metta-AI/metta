from typing import Optional

from softmax.orchestrator.clients.base_client import get_machine_token
from softmax.orchestrator.sweep_client import SweepClient


class CogwebClient:
    """Unified client for interacting with all Cogweb backend services."""

    _instances = {}

    def __init__(self, base_url: str, auth_token: str):
        """Initialize the Cogweb client. Note: Use get_client() factory method for cached instances."""
        self._base_url = base_url
        self._auth_token = auth_token
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

    # NOTE: Future service clients can be added here:
    # def stats_client(self) -> StatsClient:
    #     """Get the stats client for metrics and logging operations."""
    #     return self._stats_client
    #
    # def checkpoint_manager(self) -> CheckpointManager:
    #     """Get the checkpoint manager for model management."""
    #     return self._checkpoint_manager
