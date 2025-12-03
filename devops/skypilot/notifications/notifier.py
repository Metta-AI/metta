from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from tenacity import Retrying, stop_after_attempt, wait_exponential_jitter

from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.log_config import getRankAwareLogger

logger = getRankAwareLogger(__name__)


@dataclass
class NotificationConfig:
    """Configuration for a notification."""

    title: str
    description: str
    github_state: Literal["success", "failure", "error", "pending"]
    send_discord: bool = True
    send_wandb: bool = True
    send_github: bool = True


class NotificationBase(ABC):
    """Base class for all notifiers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def _validate_config(self, job_config: JobConfig) -> str | None:
        """Validate job config and return error message if invalid, None if valid."""
        pass

    @abstractmethod
    def _make_payload(self, notification: NotificationConfig, job_config: JobConfig) -> dict[str, Any]:
        """Create the notification payload."""
        pass

    @abstractmethod
    def _send(self, payload: dict[str, Any]) -> None:
        """Send the actual notification. Should raise exception on failure."""
        pass

    def send(self, notification: NotificationConfig, job_config: JobConfig) -> bool:
        """Main entry point for sending notifications."""

        # Validate configuration
        error_msg = self._validate_config(job_config)
        if error_msg:
            logger.warning(f"Skipping {self.name} notification - {error_msg}")
            return False

        # Create payload
        try:
            payload = self._make_payload(notification, job_config)
        except Exception as e:
            logger.error(f"Failed to format {self.name} notification: {e}", exc_info=True)
            return False

        # Log before sending
        logger.info(f"Sending {self.name} notification: {notification.title} - {notification.description}")

        # Send with retry
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(4),
                wait=wait_exponential_jitter(initial=2.0, max=30.0),
                reraise=True,
            ):
                with attempt:
                    self._send(payload)
            logger.info(f"Successfully sent {self.name} notification")
            return True
        except Exception as e:
            logger.error(f"{self.name} notification failed: {e}", exc_info=True)
            return False
