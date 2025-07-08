"""
Factory for creating agent components without Hydra.

This module provides a clean API for instantiating agent components
from configuration dictionaries without requiring Hydra.
"""

import logging
from typing import Any, Dict

from omegaconf import DictConfig

from metta.common.util.instantiate import instantiate

logger = logging.getLogger(__name__)


class ComponentFactory:
    """Factory for creating agent components from configuration."""

    @classmethod
    def create(cls, component_name: str, config: Any, agent_attributes: Dict[str, Any]) -> Any:
        """Create a component from configuration.

        Args:
            component_name: Name of the component
            config: Component configuration (DictConfig or dict)
            agent_attributes: Agent attributes to merge with config

        Returns:
            Instantiated component
        """
        # Prepare config with special handling for nn_params
        if isinstance(config, dict):
            config = config.copy()
        else:
            config = dict(config)

        # Ensure nn_params is DictConfig for dual access
        if (
            "nn_params" in config
            and isinstance(config["nn_params"], dict)
            and not isinstance(config["nn_params"], DictConfig)
        ):
            config["nn_params"] = DictConfig(config["nn_params"])

        # Merge with agent attributes and name
        config.update(agent_attributes)
        config["name"] = component_name

        logger.debug(f"Creating component {component_name}")

        # Use common instantiate function
        return instantiate(config)
