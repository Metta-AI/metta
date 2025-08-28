from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from metta.common.util.collections import remove_none_values
from metta.common.wandb.wandb_context import WandbConfig
from metta.setup.components.aws import AWSSetup
from metta.setup.components.observatory_key import ObservatoryKeySetup
from metta.setup.components.wandb import WandbSetup
from metta.setup.profiles import UserType


class SupportedWandbEnvOverrides(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
    )

    WANDB_ENABLED: bool | None = Field(default=None, description="Enable Weights & Biases")
    WANDB_PROJECT: str | None = Field(default=None, description="Weights & Biases project")
    WANDB_ENTITY: str | None = Field(default=None, description="Weights & Biases entity")

    def to_config_settings(self) -> dict[str, str | bool]:
        return remove_none_values(
            {
                "enabled": self.WANDB_ENABLED,
                "project": self.WANDB_PROJECT,
                "entity": self.WANDB_ENTITY,
            }
        )


supported_tool_overrides = SupportedWandbEnvOverrides()


def auto_wandb_config(run: str | None = None) -> WandbConfig:
    from metta.setup.saved_settings import get_saved_settings

    wandb_setup_module = WandbSetup()
    saved_settings = get_saved_settings()

    # Start with profile defaults
    config_dict = wandb_setup_module.to_config_settings()

    # Apply cloud user config if available
    if saved_settings.user_type == UserType.CLOUD:
        cloud_config = saved_settings.get_cloud_config()
        if cloud_config:
            if "wandb_entity" in cloud_config:
                config_dict["entity"] = cloud_config["wandb_entity"]
            if "wandb_project" in cloud_config:
                config_dict["project"] = cloud_config["wandb_project"]

    # Apply environment variable overrides (highest priority)
    config_dict.update(supported_tool_overrides.to_config_settings())

    cfg = WandbConfig(**config_dict)

    if run:
        cfg.name = run
        cfg.group = run
        cfg.run_id = run
        cfg.data_dir = f"./train_dir/{run}"

    return cfg


class SupportedObservatoryEnvOverrides(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
    )

    STATS_SERVER_ENABLED: bool | None = Field(default=None, description="If true, use the stats server")
    STATS_SERVER_URI: str | None = Field(default=None, description="Stats server URI")

    def to_config_settings(self) -> dict[str, str | None]:
        # If explicitly disabled, do not use stats server
        if self.STATS_SERVER_ENABLED is False:
            return {"stats_server_uri": None}
        # If explicitly provided, use stats server at given URI
        if self.STATS_SERVER_URI is not None:
            return {"stats_server_uri": self.STATS_SERVER_URI}
        return {}


supported_observatory_env_overrides = SupportedObservatoryEnvOverrides()


def auto_stats_server_uri() -> str | None:
    return {
        **ObservatoryKeySetup().to_config_settings(),  # type: ignore
        **supported_observatory_env_overrides.to_config_settings(),
    }.get("stats_server_uri")


class SupportedAwsEnvOverrides(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
    )

    REPLAY_DIR: str | None = Field(default=None, description="Replay directory")
    TORCH_PROFILE_DIR: str | None = Field(default=None, description="Torch profiler directory")
    CHECKPOINT_DIR: str | None = Field(default=None, description="Checkpoint directory")

    def to_config_settings(self) -> dict[str, str]:
        return remove_none_values(
            {
                "replay_dir": self.REPLAY_DIR,
                "torch_profile_dir": self.TORCH_PROFILE_DIR,
                "checkpoint_dir": self.CHECKPOINT_DIR,
            }
        )


supported_aws_env_overrides = SupportedAwsEnvOverrides()


def auto_replay_dir() -> str:
    from metta.setup.saved_settings import get_saved_settings

    aws_setup_module = AWSSetup()
    saved_settings = get_saved_settings()

    # Start with profile defaults
    config = aws_setup_module.to_config_settings()  # type: ignore

    # Apply cloud user config if available
    if saved_settings.user_type == UserType.CLOUD:
        cloud_config = saved_settings.get_cloud_config()
        if cloud_config and "s3_bucket" in cloud_config:
            config["replay_dir"] = f"s3://{cloud_config['s3_bucket']}/replays/"

    # Apply environment variable overrides (highest priority)
    config.update(supported_aws_env_overrides.to_config_settings())

    return config.get("replay_dir")


def auto_torch_profile_dir() -> str:
    """Returns profile-based torch profiler directory."""
    from metta.setup.saved_settings import get_saved_settings

    saved_settings = get_saved_settings()

    # Profile-based defaults
    if saved_settings.user_type.is_softmax:
        profile_default = "s3://softmax-public/torch_traces/"
    elif saved_settings.user_type == UserType.CLOUD:
        # Check for cloud user's S3 bucket
        cloud_config = saved_settings.get_cloud_config()
        if cloud_config and "s3_bucket" in cloud_config:
            profile_default = f"s3://{cloud_config['s3_bucket']}/torch_traces/"
        else:
            profile_default = "./train_dir/torch_traces/"
    else:
        profile_default = "./train_dir/torch_traces/"

    # Allow environment variable override
    config = {
        "torch_profile_dir": profile_default,
        **supported_aws_env_overrides.to_config_settings(),
    }

    return config.get("torch_profile_dir")


def auto_checkpoint_dir() -> str:
    """Returns profile-based checkpoint directory.

    Note: Currently all users use local checkpoints by default for performance.
    Cloud users can override with CHECKPOINT_DIR env var if they want S3.
    """
    # Everyone gets local by default - checkpoints are accessed frequently during training
    # and S3 latency could impact performance
    checkpoint_default = "./train_dir/checkpoints/"

    # Allow environment variable override for cloud users who want S3
    config = {
        "checkpoint_dir": checkpoint_default,
        **supported_aws_env_overrides.to_config_settings(),
    }

    return config.get("checkpoint_dir")
