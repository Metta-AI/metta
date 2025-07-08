"""
Factory for creating map builders without Hydra.

This module provides direct instantiation of map builders from configuration dictionaries.
"""

import logging
from typing import Any, Dict

from metta.common.util.instantiate import instantiate

logger = logging.getLogger(__name__)


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

        # Use common instantiate function
        return instantiate(config)

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
                return instantiate(processed)
            else:
                # Regular dict or top-level config, return processed version
                return processed
        elif isinstance(config, list):
            return [MapBuilderFactory._process_recursive(item, is_top_level=False) for item in config]
        else:
            return config
