"""WandB API client wrapper."""

import logging
from typing import Optional

try:
    import wandb
    from wandb import Api
except ImportError:
    wandb = None
    Api = None

logger = logging.getLogger(__name__)


class WandBClient:
    """Client wrapper for WandB API operations."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the WandB client.

        Args:
            api_key: Optional API key. If not provided, will use environment variable or login.
        """
        if Api is None:
            raise ImportError("wandb package not installed")

        self.api_key = api_key
        self.api = None
        self._initialize_api()

    def _initialize_api(self) -> None:
        """Initialize the WandB API client."""
        try:
            try:
                self.api = Api()
                _ = self.api.viewer
                logger.info("WandB API client initialized using cached credentials")
                return
            except Exception:
                logger.info("Cached credentials not working, trying explicit login...")

            if self.api_key:
                wandb.login(key=self.api_key)
                self.api = Api()
                logger.info("WandB API client initialized with provided API key")
            else:
                self.api = Api()
                logger.info("WandB API client initialized (authentication may be limited)")

        except Exception as e:
            logger.error(f"Failed to initialize WandB API: {e}", exc_info=True)
            raise
