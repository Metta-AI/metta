"""
Unified configuration schema for Metta.

This defines the single source of truth for all configuration.
All components should read from this configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _find_project_root() -> Path | None:
    """Find project root by looking for common project indicators."""
    current = Path.cwd()
    while current != current.parent:
        # Look for project indicators (broader than just metta repo)
        project_indicators = [
            "config.yaml",  # Metta config file
            "pyproject.toml",  # Python project
            "setup.py",  # Python project
            "requirements.txt",  # Python project
            ".git",  # Git repository
        ]

        if any((current / indicator).exists() for indicator in project_indicators):
            return current
        current = current.parent
    return None


def _get_config_path() -> Path:
    """Get configuration file path.

    Uses project root config.yaml only - no more user home complexity.
    """
    project_root = _find_project_root()

    if project_root:
        return project_root / "config.yaml"

    # If we can't find project root, assume current directory
    return Path.cwd() / "config.yaml"


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = False
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
class ProfileConfig:
    """Configuration for a single profile."""

    wandb: WandbConfig = field(default_factory=WandbConfig)
    observatory: ObservatoryConfig = field(default_factory=ObservatoryConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    datadog: DatadogConfig = field(default_factory=DatadogConfig)


@dataclass
class MettaConfig:
    """Complete Metta configuration with multi-profile support."""

    # Multi-profile support
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)

    # Backward compatibility - direct config (deprecated but supported)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    observatory: ObservatoryConfig = field(default_factory=ObservatoryConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    datadog: DatadogConfig = field(default_factory=DatadogConfig)

    # Active profile (default profile to use)
    profile: str = "external"

    def get_active_profile(self, profile_override: str | None = None) -> ProfileConfig:
        """Get the active profile configuration.

        Priority:
        1. profile_override parameter (--profile flag)
        2. METTA_PROFILE environment variable
        3. self.profile (profile in config file)
        4. Default "external" profile
        """
        import os

        active_profile_name = profile_override or os.environ.get("METTA_PROFILE") or self.profile or "external"

        # If using multi-profile config
        if self.profiles and active_profile_name in self.profiles:
            return self.profiles[active_profile_name]

        # Backward compatibility: use direct config as "external" profile
        if active_profile_name == "external" or not self.profiles:
            return ProfileConfig(
                wandb=self.wandb, observatory=self.observatory, storage=self.storage, datadog=self.datadog
            )

        # Profile not found, create default
        return ProfileConfig()

    def set_active_profile(self, profile_name: str) -> None:
        """Set the active profile in config file."""
        self.profile = profile_name

    def add_profile(self, name: str, profile_config: ProfileConfig) -> None:
        """Add or update a profile."""
        if not self.profiles:
            self.profiles = {}
        self.profiles[name] = profile_config

    def list_profiles(self) -> list[str]:
        """List available profile names."""
        profiles = list(self.profiles.keys()) if self.profiles else []

        # Always include "external" for backward compatibility
        if "external" not in profiles:
            profiles.append("external")

        return sorted(profiles)

    @classmethod
    def load(cls, path: Path | None = None) -> "MettaConfig":
        """Load configuration from project config.yaml file.

        Args:
            path: Optional explicit path to config file. If None, uses project root config.yaml.
        """
        if path is not None:
            # Explicit path provided - load directly
            if not path.exists():
                return cls()

            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}

            profiles = {}
            if "profiles" in data:
                for profile_name, profile_data in data["profiles"].items():
                    profiles[profile_name] = ProfileConfig(
                        wandb=WandbConfig(**profile_data.get("wandb", {})),
                        observatory=ObservatoryConfig(**profile_data.get("observatory", {})),
                        storage=StorageConfig(**profile_data.get("storage", {})),
                        datadog=DatadogConfig(**profile_data.get("datadog", {})),
                    )

            return cls(
                wandb=WandbConfig(**data.get("wandb", {})),
                observatory=ObservatoryConfig(**data.get("observatory", {})),
                storage=StorageConfig(**data.get("storage", {})),
                datadog=DatadogConfig(**data.get("datadog", {})),
                profile=data.get("profile", "external"),
                profiles=profiles,
            )

        # Use project root config.yaml only
        config_path = _get_config_path()

        if not config_path.exists():
            return cls()

        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Parse profiles if they exist
        profiles = {}
        if "profiles" in data:
            for profile_name, profile_data in data["profiles"].items():
                profiles[profile_name] = ProfileConfig(
                    wandb=WandbConfig(**profile_data.get("wandb", {})),
                    observatory=ObservatoryConfig(**profile_data.get("observatory", {})),
                    storage=StorageConfig(**profile_data.get("storage", {})),
                    datadog=DatadogConfig(**profile_data.get("datadog", {})),
                )

        return cls(
            profiles=profiles,
            wandb=WandbConfig(**data.get("wandb", {})),
            observatory=ObservatoryConfig(**data.get("observatory", {})),
            storage=StorageConfig(**data.get("storage", {})),
            datadog=DatadogConfig(**data.get("datadog", {})),
            profile=data.get("profile", "external"),
        )

    def save(self, path: Path | None = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = _get_config_path()

        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, removing None values and checking for default values
        wandb_data = {k: v for k, v in self.wandb.__dict__.items() if v is not None}
        observatory_data = {k: v for k, v in self.observatory.__dict__.items() if v is not None and k != "auth_token"}
        storage_data = {k: v for k, v in self.storage.__dict__.items() if v is not None}
        datadog_data = {k: v for k, v in self.datadog.__dict__.items() if v is not None and not k.endswith("_key")}

        # Create defaults for comparison
        defaults = MettaConfig()

        data = {
            "profile": self.profile,
        }

        # Add profiles if they exist
        if self.profiles:
            profiles_data = {}
            for profile_name, profile_config in self.profiles.items():
                profile_wandb = {k: v for k, v in profile_config.wandb.__dict__.items() if v is not None}
                profile_observatory = {
                    k: v for k, v in profile_config.observatory.__dict__.items() if v is not None and k != "auth_token"
                }
                profile_storage = {k: v for k, v in profile_config.storage.__dict__.items() if v is not None}
                profile_datadog = {
                    k: v for k, v in profile_config.datadog.__dict__.items() if v is not None and not k.endswith("_key")
                }

                profiles_data[profile_name] = {}
                if profile_wandb:
                    profiles_data[profile_name]["wandb"] = profile_wandb
                if profile_observatory:
                    profiles_data[profile_name]["observatory"] = profile_observatory
                if profile_storage:
                    profiles_data[profile_name]["storage"] = profile_storage
                if profile_datadog:
                    profiles_data[profile_name]["datadog"] = profile_datadog

            if profiles_data:
                data["profiles"] = profiles_data

        # Add backward compatibility sections only if no profiles exist
        if not self.profiles:
            data.update(
                {
                    "wandb": wandb_data,
                    "observatory": observatory_data,
                    "storage": storage_data,
                    "datadog": datadog_data,
                }
            )

        # For backward compatibility mode, remove sections with only default values
        if not self.profiles:
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

        # Remove any remaining empty sections (but always keep profile)
        data = {k: v for k, v in data.items() if v or k == "profile"}

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Global config instance
_config: MettaConfig | None = None


def get_config() -> MettaConfig:
    """Get the global configuration instance."""
    import threading

    global _config
    if _config is None:
        # Use double-checked locking pattern for thread safety
        with threading.Lock():
            if _config is None:
                _config = MettaConfig.load()
    return _config


def get_profile_config(profile_override: str | None = None) -> ProfileConfig:
    """Get the active profile configuration directly."""
    config = get_config()
    return config.get_active_profile(profile_override)


def reload_config() -> MettaConfig:
    """Reload configuration from disk."""
    global _config
    _config = MettaConfig.load()
    return _config
