from __future__ import annotations

import os
from typing import (
    Any,
    Type,
    TypeVar,
    Union,
)

import boto3
import hydra
import wandb
from botocore.exceptions import ClientError, NoCredentialsError
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError

T = TypeVar("T")
class Config(BaseModel):
    """
    Pydantic-backed config base.
    - extra keys are ignored
    - you can do `MyConfig(cfg_node)` where cfg_node is a DictConfig or dict
    - .dictconfig()  → OmegaConf.DictConfig
    - .yaml()        → YAML string
    """

    class Config:
        extra = "forbid"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # allow `Config(DictConfig)` or `Config(dict)` as shorthand for .from_dictconfig(...)
        if len(args) == 1 and not kwargs and isinstance(args[0], (DictConfig, dict)):
            raw = args[0]
            data = OmegaConf.to_container(raw, resolve=True) if isinstance(raw, DictConfig) else dict(raw)
            try:
                super().__init__(**data)
            except ValidationError:
                # re-raise so traceback points here
                raise
        else:
            # normal BaseModel __init__(**kwargs)
            super().__init__(*args, **kwargs)

    @classmethod
    def from_dictconfig(cls: Type[T], cfg: Union[DictConfig, dict]) -> T:
        """
        Explicit constructor from a DictConfig or plain dict.
        """
        raw = cfg
        data = OmegaConf.to_container(raw, resolve=True) if isinstance(raw, DictConfig) else dict(raw)
        return cls.parse_obj(data)

    def dictconfig(self) -> DictConfig:
        """
        Convert this model back to an OmegaConf DictConfig.
        """
        return OmegaConf.create(self.dict())

    def yaml(self) -> str:
        """
        Render this model as a YAML string.
        """
        return OmegaConf.to_yaml(self.dictconfig())


def config_from_path(config_path: str, overrides: DictConfig = None) -> DictConfig:
    if config_path is None:
        raise ValueError("Config path cannot be None")

    env_cfg = hydra.compose(config_name=config_path)
    if config_path.startswith("/"):
        config_path = config_path[1:]
    for p in config_path.split("/")[:-1]:
        env_cfg = env_cfg[p]
    if overrides not in [None, {}]:
        if env_cfg._target_ == "mettagrid.mettagrid_env.MettaGridEnvSet":
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
    if require_aws:
        # Check that AWS is good to go.
        if not check_aws_credentials():
            print("AWS is not configured, please install:")
            print("brew install awscli")
            print("and run:")
            print("python ./devops/aws/setup_sso.py")
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
