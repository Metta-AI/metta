from __future__ import annotations

import os
from pathlib import Path
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

    # Check if config path starts with a slash and adjust
    adjusted_path = config_path[1:] if config_path.startswith("/") else config_path

    try:
        env_cfg = hydra.compose(config_name=config_path)
    except Exception as e:
        # Build a useful error message
        configs_dir = Path(os.path.join(os.getcwd(), "configs"))
        search_paths = [f"{config_path}", f"{adjusted_path}", f"configs/{config_path}", f"configs/{adjusted_path}"]

        # Check if any of the paths exist
        existing_paths = []
        for path in search_paths:
            full_path = Path(os.path.join(os.getcwd(), path))
            if full_path.exists():
                existing_paths.append(str(full_path))

        # Check for YAML files in configs directory
        yaml_files = []
        if configs_dir.exists():
            yaml_files = list(configs_dir.glob("**/*.yaml"))

        error_msg = f"Could not load configuration from path '{config_path}'. "

        if existing_paths:
            error_msg += f"These related paths exist: {existing_paths}. "
        else:
            error_msg += "None of the expected paths exist. "

        if yaml_files:
            error_msg += (
                "Available YAML files in configs directory: "
                f"{[str(f.relative_to(os.getcwd())) for f in yaml_files[:10]]}"
            )

            if len(yaml_files) > 10:
                error_msg += f" and {len(yaml_files) - 10} more."

        error_msg += f"\nOriginal error: {str(e)}"

        raise ValueError(error_msg) from e

    # When hydra loads a config, it "prefixes" the keys with the path of the config file.
    # We don't want that prefix, so we remove it.
    if config_path.startswith("/"):
        config_path = config_path[1:]

    for p in config_path.split("/")[:-1]:
        try:
            env_cfg = env_cfg[p]
        except (KeyError, AttributeError) as error:
            raise ValueError(
                f"Could not access key '{p}' in configuration. "
                f"Available keys: {list(env_cfg.keys() if hasattr(env_cfg, 'keys') else [])}"
            ) from error

    if overrides not in [None, {}]:
        if env_cfg._target_.endswith(".MettaGridEnvSet"):
            raise NotImplementedError("Cannot parse overrides when using multienv_mettagrid")
        env_cfg = OmegaConf.merge(env_cfg, overrides)

    return env_cfg


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
    if cfg.wandb.track and require_wandb:
        # Check that W&B is good to go.
        if not check_wandb_credentials():
            print("W&B is not configured, please install:")
            print("pip install wandb")
            print("and run:")
            print("wandb login")
            print("Alternatively, set WANDB_API_KEY or copy ~/.netrc from another machine that has it configured.")
            exit(1)
