"""
Unified configuration schema for Metta.

This defines the single source of truth for all configuration.
All components should read from this configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = True
    entity: str | None = None
    project: str | None = None


@dataclass
class ObservatoryConfig:
    """Observatory/Stats server configuration."""

    enabled: bool = False
    stats_server_uri: str | None = None
    auth_token: str | None = None  # Stored separately in keyring


@dataclass
class StorageConfig:
    """Storage configuration for S3 and local paths."""

    s3_bucket: str | None = None
    aws_profile: str | None = None
    replay_dir: str | None = None
    torch_profile_dir: str | None = None
    checkpoint_dir: str | None = None


@dataclass
class DatadogConfig:
    """Datadog monitoring configuration."""

    enabled: bool = False
    api_key: str | None = None  # Stored separately in keyring
    app_key: str | None = None  # Stored separately in keyring


@dataclass
class MettaConfig:
    """Complete Metta configuration."""

    wandb: WandbConfig = field(default_factory=WandbConfig)
    observatory: ObservatoryConfig = field(default_factory=ObservatoryConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    datadog: DatadogConfig = field(default_factory=DatadogConfig)

    # Configuration behavior
    ignore_env_vars: bool = False  # If True, environment variables won't override config file values

    # Profile information (not user-editable)
    profile: str = "external"

    @classmethod
    def load(cls, path: Path | None = None) -> "MettaConfig":
        """Load configuration from file."""
        if path is None:
            path = Path.home() / ".metta" / "config.yaml"

        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            wandb=WandbConfig(**data.get("wandb", {})),
            observatory=ObservatoryConfig(**data.get("observatory", {})),
            storage=StorageConfig(**data.get("storage", {})),
            datadog=DatadogConfig(**data.get("datadog", {})),
            ignore_env_vars=data.get("ignore_env_vars", False),
            profile=data.get("profile", "external"),
        )

    def save(self, path: Path | None = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = Path.home() / ".metta" / "config.yaml"

        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, removing None values and checking for default values
        wandb_data = {k: v for k, v in self.wandb.__dict__.items() if v is not None}
        observatory_data = {k: v for k, v in self.observatory.__dict__.items() if v is not None and k != "auth_token"}
        storage_data = {k: v for k, v in self.storage.__dict__.items() if v is not None}
        datadog_data = {k: v for k, v in self.datadog.__dict__.items() if v is not None and not k.endswith("_key")}

        # Create defaults for comparison
        defaults = MettaConfig()

        data = {
            "wandb": wandb_data,
            "observatory": observatory_data,
            "storage": storage_data,
            "datadog": datadog_data,
            "profile": self.profile,
        }

        # Add ignore_env_vars if it's not the default value
        if self.ignore_env_vars != defaults.ignore_env_vars:
            data["ignore_env_vars"] = self.ignore_env_vars

        # Remove sections that contain only default values
        if wandb_data == {k: v for k, v in defaults.wandb.__dict__.items() if v is not None}:
            data.pop("wandb", None)
        if observatory_data == {
            k: v for k, v in defaults.observatory.__dict__.items() if v is not None and k != "auth_token"
        }:
            data.pop("observatory", None)
        if storage_data == {k: v for k, v in defaults.storage.__dict__.items() if v is not None}:
            data.pop("storage", None)
        if datadog_data == {
            k: v for k, v in defaults.datadog.__dict__.items() if v is not None and not k.endswith("_key")
        }:
            data.pop("datadog", None)
        # Always keep profile section (it's important for identifying user type)

        # Remove any remaining empty sections
        data = {k: v for k, v in data.items() if v}

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def export_env_vars(self) -> dict[str, str]:
        """Export all configuration as environment variables."""
        env_vars = {}

        # Wandb
        if self.wandb.enabled:
            env_vars["WANDB_ENABLED"] = "true"
            if self.wandb.entity:
                env_vars["WANDB_ENTITY"] = self.wandb.entity
            if self.wandb.project:
                env_vars["WANDB_PROJECT"] = self.wandb.project
        else:
            env_vars["WANDB_ENABLED"] = "false"

        # Observatory
        if self.observatory.enabled:
            env_vars["STATS_SERVER_ENABLED"] = "true"
            if self.observatory.stats_server_uri:
                env_vars["STATS_SERVER_URI"] = self.observatory.stats_server_uri
        else:
            env_vars["STATS_SERVER_ENABLED"] = "false"

        # Storage
        if self.storage.aws_profile:
            env_vars["AWS_PROFILE"] = self.storage.aws_profile
        if self.storage.replay_dir:
            env_vars["REPLAY_DIR"] = self.storage.replay_dir
        if self.storage.torch_profile_dir:
            env_vars["TORCH_PROFILE_DIR"] = self.storage.torch_profile_dir
        if self.storage.checkpoint_dir:
            env_vars["CHECKPOINT_DIR"] = self.storage.checkpoint_dir

        # Datadog
        if self.datadog.enabled:
            env_vars["DD_ENABLED"] = "true"

        return env_vars


# Global config instance
_config: MettaConfig | None = None


def get_config() -> MettaConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = MettaConfig.load()
    return _config


def reload_config() -> MettaConfig:
    """Reload configuration from disk."""
    global _config
    _config = MettaConfig.load()
    return _config
