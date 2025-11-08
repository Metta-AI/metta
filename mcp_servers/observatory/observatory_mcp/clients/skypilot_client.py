"""Skypilot client wrapper."""

import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class SkypilotClient:
    """Client wrapper for Skypilot operations."""

    def __init__(self, url: Optional[str] = None):
        """Initialize the Skypilot client.

        Args:
            url: Optional Skypilot server URL
        """
        self.url = url
        logger.info(f"Skypilot client initialized (url={url})")

    def run_command(self, cmd: list[str], timeout: int = 30) -> tuple[str, str, int]:
        """Run a Skypilot CLI command.

        Args:
            cmd: Command and arguments as list
            timeout: Command timeout in seconds

        Returns:
            Tuple of (stdout, stderr, returncode)
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as e:
            logger.error(f"Skypilot command timed out: {cmd}")
            return "", str(e), -1
        except Exception as e:
            logger.error(f"Skypilot command failed: {cmd}, error: {e}")
            return "", str(e), -1

