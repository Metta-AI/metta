import os
from typing import Any, List, Optional, Union

import boto3
import hydra
import wandb
from botocore.exceptions import ClientError, NoCredentialsError
from omegaconf import DictConfig, ListConfig, OmegaConf


def config_from_path(
    config_path: str, overrides: Optional[Union[DictConfig, dict[str, Any]]] = None
) -> Union[DictConfig, ListConfig]:
    """
    Load a configuration from a specified path, optionally with overrides.

    Args:
        config_path: Path to the configuration file
        overrides: Optional configuration overrides to merge with the loaded config

    Returns:
        The loaded and potentially merged configuration (either DictConfig or ListConfig)
    """
    # Start with a DictConfig from hydra.compose
    env_cfg: Union[DictConfig, ListConfig] = hydra.compose(config_name=config_path)

    if config_path.startswith("/"):
        config_path = config_path[1:]

    path_components: List[str] = config_path.split("/")

    # Navigate through nested config structure
    for p in path_components[:-1]:
        # Type checking for safe access
        if isinstance(env_cfg, DictConfig):
            env_cfg = env_cfg[p]
        elif isinstance(env_cfg, ListConfig):
            # Try to convert p to int for list access, or raise a meaningful error
            try:
                idx = int(p)
                env_cfg = env_cfg[idx]
            except ValueError as err:
                raise TypeError(f"Cannot use string key '{p}' with ListConfig - must be an integer index") from err
        else:
            raise TypeError(f"Unexpected config type: {type(env_cfg)}")

    # Handle the original edge case with special error message
    if overrides is not None and overrides != {}:
        if isinstance(env_cfg, DictConfig) and env_cfg.get("_target_") == "mettagrid.mettagrid_env.MettaGridEnvSet":
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


def setup_metta_environment(cfg: DictConfig, require_aws: bool = True, require_wandb: bool = True):
    """
    Set up the environment for metta, checking for required credentials.

    Args:
        cfg: The configuration object
        require_aws: Whether to require AWS credentials
        require_wandb: Whether to require W&B credentials
    """
    if require_aws:
        # Check that AWS is ready using the robust boto3 method
        if not check_aws_credentials():
            print("AWS is not configured, please install:")
            print("brew install awscli")
            print("and run:")
            print("python ./devops/aws/setup_sso.py")
            print("Alternatively, set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your environment.")
            exit(1)

    if cfg.wandb.track and require_wandb:
        # Check that W&B is ready using the robust login method
        if not check_wandb_credentials():
            print("W&B is not configured, please install:")
            print("pip install wandb")
            print("and run:")
            print("wandb login")
            print("Alternatively, set WANDB_API_KEY or copy ~/.netrc from another machine that has it configured.")
            exit(1)
