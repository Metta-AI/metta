"""
Base configuration type with OmegaConf/Hydra integration.

This module provides the BaseConfig class that all configuration types inherit from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, cast

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    """
    Pydantic-backed config base with OmegaConf/Hydra integration.

    Features:
    - Direct instantiation from DictConfig or dict: `MyConfig(cfg_node)`
    - Conversion back to DictConfig: `.dictconfig()`
    - YAML serialization: `.yaml()`
    - Extra keys are forbidden by default (fail fast on typos)
    - Full Pydantic validation

    Usage:
        class MyConfig(BaseConfig):
            __init__ = BaseConfig.__init__  # For proper IDE support

            my_field: str
            my_number: int = 10
    """

    model_config = ConfigDict(
        extra="forbid",  # Fail on unknown fields
        validate_assignment=True,  # Validate on attribute assignment
    )

    # Sub-classes should use `__init__ = BaseConfig.__init__` for proper IDE support
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 1 and not kwargs and isinstance(args[0], (DictConfig, dict)):
            super().__init__(**self.prepare_dict(args[0]))
        else:
            # normal BaseModel __init__(**kwargs)
            super().__init__(*args, **kwargs)

    def prepare_dict(self, raw: Union[DictConfig, dict]) -> Dict[str, Any]:
        """
        Prepare a dictionary config from various input formats and validate keys.

        Note: OmegaConf.to_container automatically converts any nested ListConfig
        values to regular Python lists, which is what we want for Pydantic validation.
        """
        if isinstance(raw, ListConfig):
            raise TypeError(f"Cannot create {self.__class__.__name__} from ListConfig. Expected DictConfig or dict.")

        # This converts the entire structure to plain Python types:
        # - DictConfig -> dict
        # - ListConfig -> list (for any nested list values)
        # - Interpolations are resolved
        data = OmegaConf.to_container(raw, resolve=True) if isinstance(raw, DictConfig) else dict(raw)

        # Ensure data is a proper dict with string keys
        if isinstance(data, dict):
            assert all(isinstance(k, str) for k in data.keys()), "All dictionary keys must be strings"
            return cast(Dict[str, Any], data)
        else:
            raise TypeError("Data must be convertible to a dictionary")

    def dictconfig(self) -> DictConfig:
        """Convert this model back to an OmegaConf DictConfig."""
        return OmegaConf.create(self.model_dump())

    def yaml(self) -> str:
        """Render this model as a YAML string."""
        return OmegaConf.to_yaml(self.dictconfig())

    @classmethod
    def from_yaml(cls: Type[T], yaml_path: Union[str, Path]) -> T:
        """
        Load and validate config from a YAML file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            Validated instance of the config class
        """
        cfg = OmegaConf.load(yaml_path)
        return cls(cfg)

    @classmethod
    def from_hydra_path(
        cls: Type[T], config_path: str, overrides: Optional[Union[DictConfig, Dict[str, Any]]] = None
    ) -> T:
        """
        Load configuration from a Hydra config path with overrides.

        Args:
            config_path: Path to the configuration (e.g., "sim/simple")
            overrides: Optional overrides to apply

        Returns:
            Validated instance of the config class
        """
        cfg = config_from_path(config_path, overrides)
        return cls(cfg)

    def merge_with(self: T, overrides: Union[DictConfig, Dict[str, Any]]) -> T:
        """
        Create a new config instance with overrides applied.

        Args:
            overrides: Values to override in the config

        Returns:
            New instance with overrides applied
        """
        current = self.dictconfig()
        merged = OmegaConf.merge(current, overrides)
        return self.__class__(merged)


def config_from_path(config_path: str, overrides: Optional[Union[DictConfig, Dict[str, Any]]] = None) -> DictConfig:
    """
    Load configuration from a Hydra path with better error handling.

    Args:
        config_path: Path to the configuration
        overrides: Optional overrides to apply to the configuration

    Returns:
        The loaded configuration

    Raises:
        ValueError: If the config_path is None or if the configuration could not be loaded
        TypeError: If overrides is a ListConfig
    """
    if config_path is None:
        raise ValueError("Config path cannot be None")

    if isinstance(overrides, ListConfig):
        raise TypeError("Overrides cannot be a ListConfig. Use DictConfig or dict instead.")

    cfg = hydra.compose(config_name=config_path)

    # Ensure we got a DictConfig
    if isinstance(cfg, ListConfig):
        raise TypeError(f"Config at path '{config_path}' is a ListConfig, expected DictConfig")

    # When hydra loads a config, it "prefixes" the keys with the path of the config file.
    # We don't want that prefix, so we remove it.
    if config_path.startswith("/"):
        config_path = config_path[1:]

    for p in config_path.split("/")[:-1]:
        cfg = cfg[p]

    if overrides not in [None, {}]:
        # Allow overrides that are not in the config.
        OmegaConf.set_struct(cfg, False)
        # If overrides is a dict with list values, they'll become ListConfigs after merge
        # but that's OK - they'll be converted to lists when we create Pydantic models
        cfg = OmegaConf.merge(cfg, overrides)
        OmegaConf.set_struct(cfg, True)

    return cast(DictConfig, cfg)


class ConfigRegistry:
    """
    Registry for configuration types.

    Allows registration of config types with their Hydra config group names
    for automatic type resolution.
    """

    def __init__(self):
        self._registry: Dict[str, Type[BaseConfig]] = {}
        self._path_registry: Dict[str, Type[BaseConfig]] = {}  # For specific config paths

    def register(self, config_group: str, config_class: Type[BaseConfig]) -> None:
        """Register a config class for a Hydra config group."""
        self._registry[config_group] = config_class

    def register_path(self, config_path: str, config_class: Type[BaseConfig]) -> None:
        """Register a config class for a specific config path."""
        self._path_registry[config_path] = config_class

    def get(self, config_group: str) -> Optional[Type[BaseConfig]]:
        """Get config class for a config group."""
        return self._registry.get(config_group)

    def get_by_path(self, config_path: str) -> Optional[Type[BaseConfig]]:
        """Get config class for a specific path."""
        return self._path_registry.get(config_path)

    def validate(self, config_group: str, cfg: Union[DictConfig, Dict[str, Any]]) -> BaseConfig:
        """Validate a config against its registered type."""
        config_class = self.get(config_group)
        if not config_class:
            raise KeyError(f"No config class registered for group: {config_group}")
        return config_class(cfg)


# Global registry instance
config_registry = ConfigRegistry()
