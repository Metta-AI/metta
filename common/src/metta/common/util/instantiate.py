"""
Utility for instantiating objects from configuration without Hydra.

This module provides a clean instantiate function that dynamically imports
and instantiates classes based on a _target_ field in configuration dicts.
"""

import importlib
import logging
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def instantiate(config: Dict[str, Any] | DictConfig, *, _recursive_: bool = False, **kwargs) -> Any:
    """Instantiate an object from configuration.

    This function provides similar functionality to hydra.utils.instantiate,
    dynamically importing and instantiating classes based on the _target_ field.

    Args:
        config: Configuration dict or DictConfig with _target_ field
        _recursive_: If True, recursively instantiate nested configs with _target_ fields
        **kwargs: Additional keyword arguments to pass to the constructor

    Returns:
        Instantiated object

    Raises:
        ValueError: If _target_ field is missing
        ImportError: If module cannot be imported
        AttributeError: If class cannot be found in module
    """
    # Convert OmegaConf to dict if needed
    if hasattr(config, "_metadata"):
        config = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, DictConfig):
        # Convert to dict for easier manipulation
        config = dict(config)
    elif not isinstance(config, dict):
        raise TypeError(f"Config must be dict or DictConfig, got {type(config)}")

    # If recursive, process nested configs first
    if _recursive_:
        config = _process_recursive(config, is_top_level=True)

    # Extract target
    target = config.get("_target_")
    if not target:
        raise ValueError("Configuration missing '_target_' field")

    # Split module and class name
    parts = target.split(".")
    module_path = ".".join(parts[:-1])
    class_name = parts[-1]

    try:
        # Import module
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_path}: {e}") from e
    except AttributeError as e:
        raise AttributeError(f"Module {module_path} has no class {class_name}: {e}") from e

    # Prepare constructor arguments
    # Start with config values (excluding underscore-prefixed keys)
    init_kwargs = {k: v for k, v in config.items() if not k.startswith("_")}

    # Override with any provided kwargs
    init_kwargs.update(kwargs)

    logger.debug(f"Instantiating {target} with kwargs: {list(init_kwargs.keys())}")

    # Instantiate the class
    return cls(**init_kwargs)


def _process_recursive(config: Any, is_top_level: bool = False) -> Any:
    """Recursively process config to instantiate nested objects.

    Args:
        config: Configuration to process
        is_top_level: Whether this is the top-level config (should not be instantiated)
    """
    # Convert OmegaConf to dict/list if needed
    if hasattr(config, "_metadata"):
        config = OmegaConf.to_container(config, resolve=True)

    if isinstance(config, dict):
        # First, recursively process all children
        processed = {}
        for k, v in config.items():
            processed[k] = _process_recursive(v, is_top_level=False)

        # Then check if this dict should be instantiated
        if "_target_" in processed and not is_top_level:
            # This is a nested config that should be instantiated
            logger.debug(f"Instantiating nested config with target: {processed['_target_']}")
            return instantiate(processed, _recursive_=False)  # Don't recurse again
        else:
            # Regular dict or top-level config, return processed version
            return processed
    elif isinstance(config, list):
        return [_process_recursive(item, is_top_level=False) for item in config]
    else:
        return config
