"""
Factory for creating agent components without Hydra.

This module provides a clean API for instantiating agent components
from configuration dictionaries without requiring Hydra.
"""

import logging
from typing import Any, Dict, Type

from metta.agent.lib.action import ActionEmbedding
from metta.agent.lib.actor import MettaActorSingleHead
from metta.agent.lib.lstm import LSTM
from metta.agent.lib.nn_layer_library import Conv2d, Flatten, Linear
from metta.agent.lib.obs_token_to_box_shaper import ObsTokenToBoxShaper
from metta.agent.lib.observation_normalizer import ObservationNormalizer

logger = logging.getLogger(__name__)

# Component registry mapping target strings to classes
COMPONENT_REGISTRY: Dict[str, Type] = {
    "metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper": ObsTokenToBoxShaper,
    "metta.agent.lib.observation_normalizer.ObservationNormalizer": ObservationNormalizer,
    "metta.agent.lib.nn_layer_library.Conv2d": Conv2d,
    "metta.agent.lib.nn_layer_library.Flatten": Flatten,
    "metta.agent.lib.nn_layer_library.Linear": Linear,
    "metta.agent.lib.lstm.LSTM": LSTM,
    "metta.agent.lib.action.ActionEmbedding": ActionEmbedding,
    "metta.agent.lib.actor.MettaActorSingleHead": MettaActorSingleHead,
}


class ComponentFactory:
    """Factory for creating agent components from configuration."""

    @staticmethod
    def create(component_name: str, config: Dict[str, Any], agent_attributes: Dict[str, Any]) -> Any:
        """Create a component from configuration.

        Args:
            component_name: Name of the component
            config: Component configuration dictionary
            agent_attributes: Agent-level attributes to pass to component

        Returns:
            Instantiated component

        Raises:
            ValueError: If component target is not registered
        """
        # Extract target and remove from config
        target = config.get("_target_")
        if not target:
            raise ValueError(f"Component {component_name} missing '_target_' field")

        # Get component class
        component_cls = COMPONENT_REGISTRY.get(target)
        if not component_cls:
            raise ValueError(f"Unknown component target: {target}")

        # Prepare kwargs by merging config and agent attributes
        kwargs = {k: v for k, v in config.items() if not k.startswith("_")}
        kwargs.update(agent_attributes)

        # Special handling for sources
        if "sources" in kwargs:
            # Sources will be resolved later by MettaAgent
            pass

        logger.info(f"Creating component {component_name} with target {target}")
        return component_cls(**kwargs)

    @staticmethod
    def register(target: str, component_cls: Type) -> None:
        """Register a new component class.

        Args:
            target: Target string identifier
            component_cls: Component class to register
        """
        COMPONENT_REGISTRY[target] = component_cls
