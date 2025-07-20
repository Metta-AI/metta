"""Unified client for all Cogweb backend services."""

from typing import Optional

from metta.app_backend.sweep_client import SweepClient


class CogwebClient:
    """Unified client for interacting with all Cogweb backend services."""

    _instances = {}

    # Type hints for dynamic attributes set in __new__
    _resolved_base_url: str
    _resolved_auth_token: Optional[str]

    def __new__(cls, base_url: str = "http://localhost:8000", auth_token: Optional[str] = None):
        """
        Create or return cached CogwebClient instance to avoid redundant authentication.

        Args:
            base_url: Base URL of the API server
            auth_token: Authentication token. If None, will attempt to get machine token.
        """
        # Get machine token if no auth_token provided
        if auth_token is None:
            from metta.common.util.stats_client_cfg import get_machine_token

            auth_token = get_machine_token(base_url)

        # Use base_url and auth_token as cache key
        cache_key = (base_url.rstrip("/"), auth_token)

        # Return existing instance if available
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # Create new instance and cache it
        instance = super().__new__(cls)
        # Store resolved values for __init__
        instance._resolved_base_url = base_url.rstrip("/")
        instance._resolved_auth_token = auth_token
        cls._instances[cache_key] = instance
        return instance

    def __init__(self, base_url: str = "http://localhost:8000", auth_token: Optional[str] = None):
        """
        Initialize the Cogweb client (only called once per unique instance).

        Args:
            base_url: Base URL of the API server
            auth_token: Authentication token. If None, will attempt to get machine token.
        """
        # Skip initialization if already initialized (due to caching)
        if hasattr(self, "_initialized"):
            return

        # Create the underlying SweepClient
        self._sweep_client = SweepClient(self._resolved_base_url, self._resolved_auth_token)

        self._initialized = True

    # ========================================================================
    # Sweep Methods
    # ========================================================================

    def sweep_id(self, sweep_name: str) -> str | None:
        """Get sweep ID from centralized database.

        Args:
            sweep_name: Name of the sweep

        Returns:
            The wandb sweep ID or None if sweep doesn't exist
        """
        info = self._sweep_client.get_sweep(sweep_name)
        if info.exists:
            return info.wandb_sweep_id
        else:
            return None

    def sweep_next_run_id(self, sweep_name: str) -> str:
        """Get the next run ID for a sweep (atomic operation).

        Args:
            sweep_name: Name of the sweep

        Returns:
            The next run ID for the sweep
        """
        return self._sweep_client.get_next_run_id(sweep_name)

    def create_sweep(self, sweep_name: str, project: str, entity: str, wandb_sweep_id: str):
        """Create sweep in centralized database.

        Args:
            sweep_name: Name of the sweep
            project: Project name
            entity: Entity name
            wandb_sweep_id: WandB sweep ID

        Returns:
            SweepCreateResponse containing creation status and sweep ID
        """
        return self._sweep_client.create_sweep(sweep_name, project, entity, wandb_sweep_id)
