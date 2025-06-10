from __future__ import annotations

import os
from typing import Any, Dict, Optional, TypeVar, cast

import boto3
import hydra
import wandb
from botocore.exceptions import ClientError, NoCredentialsError
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


def check_aws_credentials() -> bool:
    """Check if valid AWS credentials are available from any source."""
    if "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
        # This check is primarily for github actions.
        return True
    try:
        sts = boto3.client("sts")
        sts.get_caller_identity()
        return True
    except (NoCredentialsError, ClientError):
        return False


def check_wandb_credentials() -> bool:
    """Check if valid W&B credentials are available."""
    if "WANDB_API_KEY" in os.environ:
        # This check is primarily for github actions.
        return True
    try:
        return wandb.login(anonymous="never", timeout=10)
    except Exception:
        return False


def setup_metta_environment(cfg: ListConfig | DictConfig, require_aws: bool = True, require_wandb: bool = True):
    if require_aws:
        # Check that AWS is good to go.
        if not check_aws_credentials():
            print("AWS is not configured, please install:")
            print("brew install awscli")
            print("and run:")
            print("aws sso login --profile softmax")
            print("Alternatively, set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your environment.")
            exit(1)
    if cfg.wandb.enabled and require_wandb:
        # Check that W&B is good to go.
        if not check_wandb_credentials():
            print("W&B is not configured, please run:")
            print("wandb login")
            print("Alternatively, set WANDB_API_KEY or copy ~/.netrc from another machine that has it configured.")
            exit(1)
