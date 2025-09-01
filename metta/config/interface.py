"""
Configuration interface for components.

This separates configuration concerns from setup/installation logic.
"""

from abc import ABC, abstractmethod
from typing import Any


class ConfigurableComponent(ABC):
    """Interface for components that can be configured."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Component name for configuration section."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass

    @abstractmethod
    def interactive_configure(self, current_config: dict[str, Any]) -> dict[str, Any]:
        """
        Interactive configuration wizard for this component.

        Args:
            current_config: Current configuration for this component

        Returns:
            Updated configuration dict
        """
        pass

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """
        Validate configuration for this component.

        Args:
            config: Configuration to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        return []

    def apply_defaults(self, config: dict[str, Any], profile: str) -> dict[str, Any]:
        """
        Apply profile-based defaults to configuration.

        Args:
            config: Current configuration
            profile: User profile (external, softmax, cloud)

        Returns:
            Configuration with defaults applied
        """
        return config
