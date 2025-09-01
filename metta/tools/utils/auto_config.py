"""
Auto-configuration utilities that read from the unified config file.

This is the bridge between the new unified config system and the existing code.
Priority order:
1. Environment variables (highest)
2. Unified config file (~/.metta/config.yaml)
3. Cloud user config (from saved_settings)
4. Profile defaults
5. Hardcoded defaults (lowest)
"""

import os
from datetime import datetime
from pathlib import Path

from metta.common.util.collections import remove_falsey
from metta.common.wandb.wandb_context import WandbConfig
from metta.setup.components.aws import AWSSetup
from metta.setup.profiles import UserType


def auto_wandb_config(run: str | None = None) -> WandbConfig:
    """Get W&B configuration with proper priority chain."""
    from metta.setup.saved_settings import get_saved_settings

    # Check if new config system exists
    config_path = Path.home() / ".metta" / "config.yaml"
    if config_path.exists():
        # Use new unified config system
        from metta.config.schema import get_config

        config = get_config()

        # Build config dict with proper priority
        config_dict = {}

        # Start with config file values
        config_dict = {
            "enabled": config.wandb.enabled,
            "entity": config.wandb.entity or "",
            "project": config.wandb.project or "",
        }

        # Override with environment variables (highest priority)
        if os.environ.get("WANDB_ENABLED") is not None:
            config_dict["enabled"] = os.environ.get("WANDB_ENABLED", "").lower() == "true"
        if os.environ.get("WANDB_ENTITY"):
            config_dict["entity"] = os.environ["WANDB_ENTITY"]
        if os.environ.get("WANDB_PROJECT"):
            config_dict["project"] = os.environ["WANDB_PROJECT"]
    else:
        # Fall back to old system for backward compatibility
        from pydantic import Field
        from pydantic_settings import BaseSettings, SettingsConfigDict

        from metta.common.util.collections import remove_none_values
        from metta.setup.components.wandb import WandbSetup

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

        wandb_setup_module = WandbSetup()
        saved_settings = get_saved_settings()
        env_overrides = SupportedWandbEnvOverrides()

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
        config_dict.update(env_overrides.to_config_settings())

    cfg = WandbConfig(**config_dict)

    if run:
        cfg.name = run
        cfg.group = run
        cfg.run_id = run
        cfg.data_dir = f"./train_dir/{run}"

    return cfg


def auto_stats_server_uri() -> str | None:
    """Get stats server URI with proper priority chain."""
    # Check environment variable first
    if os.environ.get("STATS_SERVER_ENABLED") == "false":
        return None
    if os.environ.get("STATS_SERVER_URI"):
        return os.environ["STATS_SERVER_URI"]

    # Check if new config system exists
    config_path = Path.home() / ".metta" / "config.yaml"
    if config_path.exists():
        from metta.config.schema import get_config

        config = get_config()
        if config.observatory.enabled and config.observatory.stats_server_uri:
            return config.observatory.stats_server_uri
    else:
        # Fall back to old system
        from metta.setup.components.observatory_key import ObservatoryKeySetup

        return ObservatoryKeySetup().to_config_settings().get("stats_server_uri")

    return None


def auto_replay_dir() -> str:
    """Get replay directory with proper priority chain."""
    from metta.setup.saved_settings import get_saved_settings

    # Check environment variable first
    if os.environ.get("REPLAY_DIR"):
        return os.environ["REPLAY_DIR"]

    # Check if new config system exists
    config_path = Path.home() / ".metta" / "config.yaml"
    if config_path.exists():
        from metta.config.schema import get_config

        config = get_config()
        if config.storage.replay_dir:
            return config.storage.replay_dir
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

        return config.get("replay_dir")

    # Default
    return "./train_dir/replays/"


def auto_torch_profile_dir() -> str:
    """Get torch profiler directory with proper priority chain."""
    from metta.setup.saved_settings import get_saved_settings

    # Check environment variable first
    if os.environ.get("TORCH_PROFILE_DIR"):
        return os.environ["TORCH_PROFILE_DIR"]

    # Check if new config system exists
    config_path = Path.home() / ".metta" / "config.yaml"
    if config_path.exists():
        from metta.config.schema import get_config

        config = get_config()
        if config.storage.torch_profile_dir:
            return config.storage.torch_profile_dir

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
    # Check environment variable first
    if os.environ.get("CHECKPOINT_DIR"):
        return os.environ["CHECKPOINT_DIR"]

    # Check if new config system exists
    config_path = Path.home() / ".metta" / "config.yaml"
    if config_path.exists():
        from metta.config.schema import get_config

        config = get_config()
        if config.storage.checkpoint_dir:
            return config.storage.checkpoint_dir

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
