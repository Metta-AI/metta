"""
Utility for instantiating objects from configuration without Hydra.

This module provides a clean instantiate function that dynamically imports
and instantiates classes based on a _target_ field in configuration dicts.
"""

import importlib
import logging
from typing import Any, Dict

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def instantiate(config: Dict[str, Any] | DictConfig, **kwargs) -> Any:
    """Instantiate an object from configuration.

    This function provides similar functionality to hydra.utils.instantiate,
    dynamically importing and instantiating classes based on the _target_ field.

    Args:
        config: Configuration dict or DictConfig with _target_ field
        **kwargs: Additional keyword arguments to pass to the constructor

    Returns:
        Instantiated object

    Raises:
        ValueError: If _target_ field is missing
        ImportError: If module cannot be imported
        AttributeError: If class cannot be found in module
    """
    if isinstance(config, DictConfig):
        # Convert to dict for easier manipulation
        config = dict(config)
    elif not isinstance(config, dict):
        raise TypeError(f"Config must be dict or DictConfig, got {type(config)}")

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
        raise ImportError(f"Failed to import module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Module {module_path} has no class {class_name}: {e}")

    # Prepare constructor arguments
    # Start with config values (excluding underscore-prefixed keys)
    init_kwargs = {k: v for k, v in config.items() if not k.startswith("_")}

    # Override with any provided kwargs
    init_kwargs.update(kwargs)

    logger.debug(f"Instantiating {target} with kwargs: {list(init_kwargs.keys())}")

    # Instantiate the class
    return cls(**init_kwargs)
