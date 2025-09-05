"""
Auto-configuration utilities that read from the unified config file.

This is the bridge between the new unified config system and the existing code.
Priority order:
1. Unified config file (project_root/config.yaml) (highest)
2. Cloud user config (from saved_settings)
3. Profile defaults
4. Hardcoded defaults (lowest)

The config file path is determined by _get_config_path() which finds the project root
and uses config.yaml from there.
"""

import os
from datetime import datetime

from metta.common.util.collections import remove_falsey
from metta.common.wandb.wandb_context import WandbConfig
from metta.setup.components.aws import AWSSetup
from metta.setup.profiles import UserType


def auto_wandb_config(run: str | None = None) -> WandbConfig:
    """Get W&B configuration with proper priority chain."""
    # Check if config system exists
    from metta.config.schema import _get_config_path
    from metta.setup.saved_settings import get_saved_settings

    config_path = _get_config_path()
    if config_path.exists():
        # Use new unified config system with active profile
        from pydantic import Field
        from pydantic_settings import BaseSettings, SettingsConfigDict

        from metta.common.util.collections import remove_none_values
        from metta.config.schema import get_config

        class SupportedWandbEnvOverrides(BaseSettings):
            model_config = SettingsConfigDict(extra="ignore")
            WANDB_ENABLED: bool | None = Field(default=None)
            WANDB_PROJECT: str | None = Field(default=None)
            WANDB_ENTITY: str | None = Field(default=None)

            def to_config_settings(self) -> dict[str, str | bool]:
                return remove_none_values(
                    {
                        "enabled": self.WANDB_ENABLED,
                        "project": self.WANDB_PROJECT,
                        "entity": self.WANDB_ENTITY,
                    }
                )

        config = get_config()
        profile_config = config.get_active_profile()
        saved_settings = get_saved_settings()

        # Start with profile config from config file
        config_dict = {
            "enabled": profile_config.wandb.enabled,
            "entity": profile_config.wandb.entity,
            "project": profile_config.wandb.project,
        }

        # Apply cloud user config if available (for backward compatibility)
        if saved_settings.user_type == UserType.CLOUD:
            cloud_config = saved_settings.get_cloud_config()
            if cloud_config:
                if "wandb_entity" in cloud_config:
                    config_dict["entity"] = cloud_config["wandb_entity"]
                if "wandb_project" in cloud_config:
                    config_dict["project"] = cloud_config["wandb_project"]

        # Apply environment variable fallbacks (only if config values are None/empty)
        supported_tool_override = SupportedWandbEnvOverrides()
        env_overrides = supported_tool_override.to_config_settings()
        for key, value in env_overrides.items():
            if config_dict.get(key) is None:  # Only use env var if config value is None
                config_dict[key] = value

        # Use empty string defaults for None values to match old behavior
        if config_dict["entity"] is None:
            config_dict["entity"] = ""
        if config_dict["project"] is None:
            config_dict["project"] = ""
    else:
        # Fall back to old system for backward compatibility
        from metta.setup.components.wandb import WandbSetup

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

        # Apply environment variable overrides (for backward compatibility)
        from pydantic import Field
        from pydantic_settings import BaseSettings, SettingsConfigDict

        from metta.common.util.collections import remove_none_values

        class SupportedWandbEnvOverrides(BaseSettings):
            model_config = SettingsConfigDict(extra="ignore")
            WANDB_ENABLED: bool | None = Field(default=None)
            WANDB_PROJECT: str | None = Field(default=None)
            WANDB_ENTITY: str | None = Field(default=None)

            def to_config_settings(self) -> dict[str, str | bool]:
                return remove_none_values(
                    {
                        "enabled": self.WANDB_ENABLED,
                        "project": self.WANDB_PROJECT,
                        "entity": self.WANDB_ENTITY,
                    }
                )

        supported_tool_override = SupportedWandbEnvOverrides()
        env_overrides = supported_tool_override.to_config_settings()
        config_dict.update(env_overrides)

    cfg = WandbConfig(**config_dict)

    if run:
        cfg.run_id = run
        cfg.group = run
        cfg.data_dir = f"./train_dir/{run}"

    return cfg


def auto_stats_server_uri() -> str | None:
    """Get stats server URI with proper priority chain."""
    # Check if config system exists
    from metta.config.schema import _get_config_path

    config_path = _get_config_path()
    if config_path.exists():
        from metta.config.schema import get_config

        config = get_config()
        profile_config = config.get_active_profile()

        # Use active profile configuration
        result = None
        if profile_config.observatory.enabled and profile_config.observatory.stats_server_uri:
            result = profile_config.observatory.stats_server_uri

        # Config file takes precedence - environment variables only used as fallback

        return result
    else:
        # Fall back to old system
        from metta.setup.components.observatory_key import ObservatoryKeySetup

        return ObservatoryKeySetup().to_config_settings().get("stats_server_uri")

    return None


def auto_replay_dir() -> str:
    """Get replay directory with proper priority chain."""
    # Check if config system exists
    from metta.config.schema import _get_config_path
    from metta.setup.saved_settings import get_saved_settings

    config_path = _get_config_path()
    if config_path.exists():
        from metta.config.schema import get_config

        config = get_config()
        profile_config = config.get_active_profile()

        # Use active profile configuration
        result = profile_config.storage.replay_dir

        # Config file takes precedence - environment variables only used as fallback
        # Environment variables can still be used if no config value is set
        if not result and os.environ.get("REPLAY_DIR"):
            result = os.environ["REPLAY_DIR"]

        if result:
            return result
    else:
        # Fall back to old system for backward compatibility
        aws_setup_module = AWSSetup()
        saved_settings = get_saved_settings()

        # Start with profile defaults
        config = aws_setup_module.to_config_settings()  # type: ignore

        # Apply cloud user config if available
        if saved_settings.user_type == UserType.CLOUD:
            cloud_config = saved_settings.get_cloud_config()
            if cloud_config and "s3_bucket" in cloud_config:
                config["replay_dir"] = f"s3://{cloud_config['s3_bucket']}/replays/"

        if "replay_dir" in config:
            return config["replay_dir"]

    # Default
    return "./train_dir/replays/"


def auto_torch_profile_dir() -> str:
    """Get torch profiler directory with proper priority chain."""
    # Check if config system exists
    from metta.config.schema import _get_config_path
    from metta.setup.saved_settings import get_saved_settings

    config_path = _get_config_path()
    if config_path.exists():
        from metta.config.schema import get_config

        config = get_config()
        profile_config = config.get_active_profile()

        # Use active profile configuration
        result = profile_config.storage.torch_profile_dir

        # Environment variables can still be used if no config value is set
        if not result and os.environ.get("TORCH_PROFILE_DIR"):
            result = os.environ["TORCH_PROFILE_DIR"]

        if result:
            return result

    # Fall back to profile-based defaults (for backward compatibility)
    saved_settings = get_saved_settings()

    # Profile-based defaults
    if saved_settings.user_type.is_softmax:
        return "s3://softmax-public/torch_traces/"
    elif saved_settings.user_type == UserType.CLOUD:
        # Check for cloud user's S3 bucket
        cloud_config = saved_settings.get_cloud_config()
        if cloud_config and "s3_bucket" in cloud_config:
            return f"s3://{cloud_config['s3_bucket']}/torch_traces/"

    # Default
    return "./train_dir/torch_traces/"


def auto_checkpoint_dir() -> str:
    """Get checkpoint directory with proper priority chain."""
    # Check if config system exists
    from metta.config.schema import _get_config_path

    config_path = _get_config_path()
    if config_path.exists():
        from metta.config.schema import get_config

        config = get_config()
        profile_config = config.get_active_profile()

        # Use active profile configuration
        result = profile_config.storage.checkpoint_dir

        # Environment variables can still be used if no config value is set
        if not result and os.environ.get("CHECKPOINT_DIR"):
            result = os.environ["CHECKPOINT_DIR"]

        if result:
            return result

    # Default (always local for performance)
    return "./train_dir/checkpoints/"


def auto_run_name(prefix: str | None = None) -> str:
    """Generate an automatic run name."""
    return ".".join(
        remove_falsey(
            [
                prefix,
                os.getenv("USER", "unknown"),
                datetime.now().strftime("%Y%m%d.%H%M%S"),
            ]
        )
    )
