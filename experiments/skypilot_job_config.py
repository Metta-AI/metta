"""Skypilot job infrastructure configuration."""

from metta.common.util.config import Config


class SkypilotJobConfig(Config):
    """Configuration for Skypilot job infrastructure.

    This contains only the core Skypilot settings.
    Other options like timeouts are handled by our wrapper scripts.
    """

    # Core Skypilot resource settings
    gpus: int = 1
    nodes: int = 1
    spot: bool = True  # Use spot instances to save costs

    # Launch control
    git_check: bool = True
