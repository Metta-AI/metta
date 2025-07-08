"""
Factory for creating agent components without Hydra.

This module provides a clean API for instantiating agent components
from configuration dictionaries without requiring Hydra.
"""

import logging
from typing import Any, Dict, Type

from omegaconf import DictConfig

from metta.agent.lib.action import ActionEmbedding
from metta.agent.lib.actor import MettaActorSingleHead
from metta.agent.lib.lstm import LSTM
from metta.agent.lib.nn_layer_library import Conv2d, Flatten, Linear
from metta.agent.lib.obs_token_to_box_shaper import ObsTokenToBoxShaper
from metta.agent.lib.observation_normalizer import ObservationNormalizer

logger = logging.getLogger(__name__)


class DictNamespace:
    """A namespace object that supports both dict-style and attribute access."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()


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
        # Get the target class
        target = config.get("_target_", "")
        component_cls = cls._get_registry().get(target)

        if not component_cls:
            raise ValueError(f"Unknown component target: {target}")

        # Prepare kwargs by merging config and agent_attributes
        kwargs = {}

        # Add all non-underscore config keys
        for k, v in config.items():
            if not k.startswith("_"):
                # If config is already DictConfig, just pass values through
                # Components expect nn_params as DictConfig for dual dict/attribute access
                if k == "nn_params" and isinstance(v, dict) and not isinstance(v, DictConfig):
                    kwargs[k] = DictConfig(v)
                else:
                    kwargs[k] = v

        # Add agent attributes
        kwargs.update(agent_attributes)

        # Add the name
        kwargs["name"] = component_name

        logger.debug(f"Creating component {component_name} with kwargs: {list(kwargs.keys())}")

        return component_cls(**kwargs)

    @staticmethod
    def register(target: str, component_cls: Type) -> None:
        """Register a new component class.

        Args:
            target: Target string identifier
            component_cls: Component class to register
        """
        COMPONENT_REGISTRY[target] = component_cls

    @classmethod
    def _get_registry(cls) -> Dict[str, Type]:
        return COMPONENT_REGISTRY
