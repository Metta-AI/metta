"""
BuildContext for capturing how policies were constructed.

This module provides BuildContext which captures all the information needed
to reconstruct a policy, including configuration, source code, and environment
attributes. This is particularly important for BrainPolicy which builds
complex networks from YAML configuration.
"""

from typing import Any, Dict, Optional, Tuple


class BuildContext:
    """Captures how a policy was built for later reconstruction."""

    def __init__(
        self,
        method: str,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        env_attributes: Optional[Dict[str, Any]] = None,
        source_code: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        component_configs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize build context.

        Args:
            method: The builder method used (e.g., "build_from_brain_policy")
            args: Positional arguments passed to the builder
            kwargs: Keyword arguments passed to the builder
            env_attributes: Environment attributes needed for reconstruction
            source_code: Dictionary mapping class paths to source code
            config: The full configuration used (particularly for BrainPolicy)
            component_configs: Component-specific configurations for BrainPolicy
        """
        self.method = method
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.env_attributes = env_attributes or {}
        self.source_code = source_code or {}
        self.config = config or {}
        self.component_configs = component_configs or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method,
            "args": self.args,
            "kwargs": self.kwargs,
            "env_attributes": self.env_attributes,
            "source_code": self.source_code,
            "config": self.config,
            "component_configs": self.component_configs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildContext":
        """Create from dictionary."""
        return cls(
            method=data.get("method", ""),
            args=tuple(data.get("args", [])),
            kwargs=data.get("kwargs", {}),
            env_attributes=data.get("env_attributes", {}),
            source_code=data.get("source_code", {}),
            config=data.get("config", {}),
            component_configs=data.get("component_configs", {}),
        )
