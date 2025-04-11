from typing import Any, List, Optional, Union

import hydra
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
                raise TypeError(
                    f"Cannot use string key '{p}' with ListConfig - must be an integer index"
                ) from err
        else:
            raise TypeError(f"Unexpected config type: {type(env_cfg)}")

    # Apply overrides if provided
    if overrides is not None:
        env_cfg = OmegaConf.merge(env_cfg, overrides)

    return env_cfg


def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def setup_metta_environment(cfg: DictConfig, require_aws: bool = True, require_wandb: bool = True):
    if require_aws:
        # Check that AWS is good to go.
        # Check that ~/.aws/credentials exist or env var AWS_PROFILE is set.
        if (
            not os.path.exists(os.path.expanduser("~/.aws/sso/cache"))
            and "AWS_ACCESS_KEY_ID" not in os.environ
            and "AWS_SECRET_ACCESS_KEY" not in os.environ
        ):
            print("AWS is not configured, please install:")
            print("brew install awscli")
            print("and run:")
            print("python ./devops/aws/setup_sso.py")
            print(
                "Alternatively, set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your environment."
            )
            exit(1)
    if cfg.wandb.track and require_wandb:
        # Check that W&B is good to go.
        # Open ~/.netrc file and see if there is a api.wandb.ai entry.
        if (
            "api.wandb.ai" not in read_file(os.path.expanduser("~/.netrc"))
            and "WANDB_API_KEY" not in os.environ
        ):
            print("W&B is not configured, please install:")
            print("pip install wandb")
            print("and run:")
            print("wandb login")
            print(
                "Alternatively, set WANDB_API_KEY or copy ~/.netrc from another machine that has it configured."
            )
            exit(1)
