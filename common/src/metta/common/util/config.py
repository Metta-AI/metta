from __future__ import annotations

from typing import Any, Dict, Optional, TypeVar, cast

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel

T = TypeVar("T")


class Config(BaseModel):
    """
    Pydantic-backed config base.
    - extra keys are ignored
    - you can do `MyConfig(cfg_node)` where cfg_node is a DictConfig or dict
    - .dictconfig() → OmegaConf.DictConfig
    - .yaml() → YAML string
    """

    model_config = {"extra": "forbid"}

    # Sub-classes of Config class should use the `__init__ = Config.__init__` trick to satisfy Pylance.
    # Without this, Pylance will complain about 0 positional arguments, because it looks up Pydantic's
    # __init__ method, which takes no positional arguments.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 1 and not kwargs and isinstance(args[0], (DictConfig, dict)):
            super().__init__(**self.prepare_dict(args[0]))
        else:
            # normal BaseModel __init__(**kwargs)
            super().__init__(*args, **kwargs)

    def dictconfig(self) -> DictConfig:
        """
        Convert this model back to an OmegaConf DictConfig.
        """
        # Use model_dump() in Pydantic v2
        return OmegaConf.create(self.model_dump())

    def yaml(self) -> str:
        """
        Render this model as a YAML string.
        """
        return OmegaConf.to_yaml(self.dictconfig())

    def prepare_dict(self, raw) -> Dict[str, Any]:
        """Prepare a dictionary config from various input formats and validate keys."""
        data = OmegaConf.to_container(raw, resolve=True) if isinstance(raw, DictConfig) else dict(raw)
        # Ensure data is a proper dict with string keys
        if isinstance(data, dict):
            assert all(isinstance(k, str) for k in data.keys()), "All dictionary keys must be strings"
            return cast(Dict[str, Any], data)
        else:
            raise TypeError("Data must be convertible to a dictionary")


def config_from_path(config_path: str, overrides: Optional[DictConfig | ListConfig] = None) -> DictConfig | ListConfig:
    """
    Load configuration from a path, with better error handling

    Args:
        config_path: Path to the configuration
        overrides: Optional overrides to apply to the configuration

    Returns:
        The loaded configuration

    Raises:
        ValueError: If the config_path is None or if the configuration could not be loaded
    """
    if config_path is None:
        raise ValueError("Config path cannot be None")

    cfg = hydra.compose(config_name=config_path)

    # when hydra loads a config, it "prefixes" the keys with the path of the config file.
    # We don't want that prefix, so we remove it.
    if config_path.startswith("/"):
        config_path = config_path[1:]

    for p in config_path.split("/")[:-1]:
        cfg = cfg[p]

    if overrides not in [None, {}]:
        # Allow overrides that are not in the config.
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, overrides)
        OmegaConf.set_struct(cfg, True)
    return cast(DictConfig, cfg)


def copy_omegaconf_config(cfg: DictConfig | ListConfig) -> DictConfig | ListConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
