"""
Factory for creating map builders without Hydra.

This module provides direct instantiation of map builders from configuration dictionaries.
"""

import logging
from typing import Any, Dict, Type

from metta.mettagrid.room.ascii import Ascii
from metta.mettagrid.room.maze import MazeKruskal, MazePrim
from metta.mettagrid.room.mean_distance import MeanDistance
from metta.mettagrid.room.multi_room import MultiRoom
from metta.mettagrid.room.random import Random
from metta.mettagrid.room.room_list import RoomList
from metta.mettagrid.room.terrain_from_numpy import TerrainFromNumpy

logger = logging.getLogger(__name__)

# Map builder registry
MAP_BUILDER_REGISTRY: Dict[str, Type] = {
    "metta.mettagrid.room.ascii.Ascii": Ascii,
    "metta.mettagrid.room.maze.MazeKruskal": MazeKruskal,
    "metta.mettagrid.room.maze.MazePrim": MazePrim,
    "metta.mettagrid.room.mean_distance.MeanDistance": MeanDistance,
    "metta.mettagrid.room.multi_room.MultiRoom": MultiRoom,
    "metta.mettagrid.room.random.Random": Random,
    "metta.mettagrid.room.room_list.RoomList": RoomList,
    "metta.mettagrid.room.terrain_from_numpy.TerrainFromNumpy": TerrainFromNumpy,
}


class MapBuilderFactory:
    """Factory for creating map builders from configuration."""

    @staticmethod
    def create(config: Dict[str, Any], recursive: bool = True) -> Any:
        """Create a map builder from configuration.

        Args:
            config: Map builder configuration dictionary or OmegaConf object
            recursive: Whether to recursively instantiate nested configs

        Returns:
            Instantiated map builder

        Raises:
            ValueError: If target is not registered
        """
        # Convert OmegaConf to dict if needed
        if hasattr(config, "_metadata"):
            from omegaconf import OmegaConf

            config = OmegaConf.to_container(config, resolve=True)

        # Handle nested configs recursively
        if recursive:
            # Process all nested configs first (but don't instantiate the top level)
            config = MapBuilderFactory._process_recursive(config, is_top_level=True)

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
    def _process_recursive(config: Any, is_top_level: bool = False) -> Any:
        """Recursively process config to instantiate nested builders.

        Args:
            config: Configuration to process
            is_top_level: Whether this is the top-level config (should not be instantiated)
        """
        # Convert OmegaConf to dict/list if needed
        if hasattr(config, "_metadata"):
            from omegaconf import OmegaConf

            config = OmegaConf.to_container(config, resolve=True)

        if isinstance(config, dict):
            # First, recursively process all children
            processed = {}
            for k, v in config.items():
                processed[k] = MapBuilderFactory._process_recursive(v, is_top_level=False)

            # Then check if this dict should be instantiated
            if "_target_" in processed and not is_top_level:
                # This is a nested config that should be instantiated
                logger.debug(f"Instantiating nested config with target: {processed['_target_']}")
                return MapBuilderFactory.create(processed, recursive=False)
            else:
                # Regular dict or top-level config, return processed version
                return processed
        elif isinstance(config, list):
            return [MapBuilderFactory._process_recursive(item, is_top_level=False) for item in config]
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
