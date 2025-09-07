"""
Tribal Environment Configuration for Metta Training Infrastructure.

This module provides a configuration bridge between the Nim tribal environment
and the existing MettaGrid-based training system.
"""

from typing import Optional

from metta.mettagrid.config import Config
from metta.mettagrid.mettagrid_config import MettaGridConfig


class TribalConfig(Config):
    """Configuration wrapper for tribal environment."""

    # Basic environment parameters
    num_agents: int = 15
    max_steps: int = 2000
    render_mode: Optional[str] = None

    # Environment-specific settings
    environment_type: str = "tribal"
    bindings_path: Optional[str] = None

    def to_mettagrid_config(self) -> "TribalMettaGridConfig":
        """Convert to MettaGridConfig-compatible format."""
        return TribalMettaGridConfig(
            num_agents=self.num_agents,
            max_steps=self.max_steps,
            render_mode=self.render_mode,
            environment_type=self.environment_type,
            bindings_path=self.bindings_path,
        )


class TribalMettaGridConfig(MettaGridConfig):
    """
    Tribal environment config that inherits from MettaGridConfig.

    This allows tribal environments to work with the existing training infrastructure
    while maintaining their unique characteristics.
    """

    # Override the environment type
    environment_type: str = "tribal"
    max_steps: int = 2000
    render_mode: Optional[str] = None
    bindings_path: Optional[str] = None

    def __init__(self, **kwargs):
        # Set default values for required MettaGridConfig fields
        defaults = {
            # Basic game configuration - dummy values for tribal
            "game": {
                "agent": {
                    "default_resource_limit": 255,
                    "resource_limits": {},
                    "freeze_duration": 10,
                    "rewards": {"inventory": {}, "inventory_max": {}, "stats": {}},
                    "action_failure_penalty": 0.0,
                    "initial_inventory": {},
                    "team_id": 0,
                },
                "actions": {
                    "attack": {"enabled": True, "consumed_resources": {}},
                    "get_items": {"enabled": True},
                    "move": {"enabled": True},
                    "noop": {"enabled": True},
                    "put_items": {"enabled": True},
                },
                "objects": {},
                "resource_loss_prob": 0.0,
                "map_builder": {"type": "tribal_custom", "num_agents": kwargs.get("num_agents", 15)},
            }
        }

        # Merge with provided kwargs
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs[key] = value

        super().__init__(**kwargs)


def make_tribal_config(num_agents: int = 15, max_steps: int = 2000, **kwargs) -> TribalMettaGridConfig:
    """
    Create a tribal environment configuration compatible with MettaGrid training.

    Args:
        num_agents: Number of agents in the environment
        max_steps: Maximum steps per episode
        **kwargs: Additional configuration parameters

    Returns:
        TribalMettaGridConfig instance
    """
    return TribalMettaGridConfig(num_agents=num_agents, max_steps=max_steps, environment_type="tribal", **kwargs)
