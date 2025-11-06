import abc
import dataclasses
import typing

import devops.skypilot.utils.job_config
import metta.common.util.log_config
import metta.common.util.retry

logger = metta.common.util.log_config.getRankAwareLogger(__name__)


@dataclasses.dataclass
class NotificationConfig:
    """Configuration for a notification."""

    title: str
    description: str
    github_state: typing.Literal["success", "failure", "error", "pending"]
    send_discord: bool = True
    send_wandb: bool = True
    send_github: bool = True


class NotificationBase(abc.ABC):
    """Base class for all notifiers."""

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def _validate_config(self, job_config: devops.skypilot.utils.job_config.JobConfig) -> str | None:
        """Validate job config and return error message if invalid, None if valid."""
        pass

    @abc.abstractmethod
    def _make_payload(
        self, notification: NotificationConfig, job_config: devops.skypilot.utils.job_config.JobConfig
    ) -> dict[str, typing.Any]:
        """Create the notification payload."""
        pass

    @abc.abstractmethod
    def _send(self, payload: dict[str, typing.Any]) -> None:
        """Send the actual notification. Should raise exception on failure."""
        pass

    def send(self, notification: NotificationConfig, job_config: devops.skypilot.utils.job_config.JobConfig) -> bool:
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
            metta.common.util.retry.retry_function(
                lambda: self._send(payload),
                max_retries=3,
                initial_delay=2.0,
                max_delay=30.0,
            )
            logger.info(f"✅ Successfully sent {self.name} notification")
            return True
        except Exception as e:
            logger.error(f"{self.name} notification failed: {e}", exc_info=True)
            return False
