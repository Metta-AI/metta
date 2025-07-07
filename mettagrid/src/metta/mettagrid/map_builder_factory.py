"""
Factory for creating map builders without Hydra.

This module provides direct instantiation of map builders from configuration dictionaries.
"""

import logging
from typing import Any, Dict, Type

from metta.mettagrid.room.multi_room import MultiRoom
from metta.mettagrid.room.terrain_from_numpy import TerrainFromNumpy

logger = logging.getLogger(__name__)

# Map builder registry
MAP_BUILDER_REGISTRY: Dict[str, Type] = {
    "metta.mettagrid.room.multi_room.MultiRoom": MultiRoom,
    "metta.mettagrid.room.terrain_from_numpy.TerrainFromNumpy": TerrainFromNumpy,
}


class MapBuilderFactory:
    """Factory for creating map builders from configuration."""

    @staticmethod
    def create(config: Dict[str, Any], recursive: bool = True) -> Any:
        """Create a map builder from configuration.

        Args:
            config: Map builder configuration dictionary
            recursive: Whether to recursively instantiate nested configs

        Returns:
            Instantiated map builder

        Raises:
            ValueError: If target is not registered
        """
        # Handle nested configs recursively
        if recursive:
            config = MapBuilderFactory._process_recursive(config)

        # Extract target
        target = config.get("_target_")
        if not target:
            raise ValueError("Map builder config missing '_target_' field")

        # Get builder class
        builder_cls = MAP_BUILDER_REGISTRY.get(target)
        if not builder_cls:
            raise ValueError(f"Unknown map builder target: {target}")

        # Prepare kwargs
        kwargs = {k: v for k, v in config.items() if not k.startswith("_")}

        logger.info(f"Creating map builder with target {target}")
        return builder_cls(**kwargs)

    @staticmethod
    def _process_recursive(config: Any) -> Any:
        """Recursively process config to instantiate nested builders."""
        if isinstance(config, dict):
            # Check if this dict should be instantiated
            if "_target_" in config:
                # Don't instantiate the top level, but process children first
                processed = {}
                for k, v in config.items():
                    if not k.startswith("_"):
                        processed[k] = MapBuilderFactory._process_recursive(v)
                    else:
                        processed[k] = v

                # If this is a nested config (not the top level), instantiate it
                if processed != config:
                    config = processed
                return config
            else:
                # Regular dict, process children
                return {k: MapBuilderFactory._process_recursive(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [MapBuilderFactory._process_recursive(item) for item in config]
        else:
            return config

    @staticmethod
    def register(target: str, builder_cls: Type) -> None:
        """Register a new map builder class.

        Args:
            target: Target string identifier
            builder_cls: Map builder class to register
        """
        MAP_BUILDER_REGISTRY[target] = builder_cls
