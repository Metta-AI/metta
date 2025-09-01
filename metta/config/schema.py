"""
Unified configuration schema for Metta.

This defines the single source of truth for all configuration.
All components should read from this configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _get_profiles_dir() -> Path:
    """Get the profiles directory, supporting environment variable override."""
    import os

    # Check for environment variable override (like DBT_PROFILES_DIR)
    env_profiles_dir = os.environ.get("METTA_PROFILES_DIR")
    if env_profiles_dir:
        return Path(env_profiles_dir)

    # Default to ~/.metta (like ~/.dbt)
    return Path.home() / ".metta"


def _find_project_root() -> Path | None:
    """Find project root by looking for common project indicators."""
    current = Path.cwd()
    while current != current.parent:
        # Look for project indicators (broader than just metta repo)
        project_indicators = [
            "pyproject.toml",  # Python project
            "setup.py",  # Python project
            "requirements.txt",  # Python project
            ".git",  # Git repository
        ]

        if any((current / indicator).exists() for indicator in project_indicators):
            return current
        current = current.parent
    return None


def _get_config_paths() -> tuple[Path | None, Path]:
    """Get configuration paths using simple priority system.

    Returns:
        Tuple of (project_config_path, global_config_path)
    """
    project_root = _find_project_root()
    profiles_dir = _get_profiles_dir()

    project_config_path = None
    if project_root:
        project_config_path = project_root / ".metta" / "config.yaml"

    global_config_path = profiles_dir / "config.yaml"

    return project_config_path, global_config_path


def _get_config_path() -> Path:
    """Get configuration file path for saving new configs.

    Uses priority system:
    1. Project config: .metta/config.yaml in project root
    2. Global profiles: ~/.metta/config.yaml (or METTA_PROFILES_DIR)
    """
    project_config_path, global_config_path = _get_config_paths()

    # Check which config exists
    if project_config_path and project_config_path.exists():
        return project_config_path
    if global_config_path.exists():
        return global_config_path

    # For new configs: prefer project location if in project and real environment
    # This prevents project config creation during tests when Path.home() is mocked
    if project_config_path and Path.home() == Path("~").expanduser():
        return project_config_path

    # Fall back to profiles directory
    return global_config_path


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
        """Load configuration using simple priority system.

        Priority order:
        1. Project config (.metta/config.yaml in project root)
        2. Global profiles config (~/.metta/config.yaml)
        3. Default values
        """
        if path is not None:
            # Explicit path provided - load directly
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

        # Use simple priority system with config.yaml files only
        project_config_path, global_config_path = _get_config_paths()

        # Start with defaults
        config_data = {}

        # 1. Load global profiles config first (lower priority)
        if global_config_path.exists():
            with open(global_config_path, "r") as f:
                global_data = yaml.safe_load(f) or {}
                config_data.update(global_data)

        # 2. Load project config (higher priority, overrides global)
        if project_config_path and project_config_path.exists():
            with open(project_config_path, "r") as f:
                project_data = yaml.safe_load(f) or {}
                # Deep merge the dictionaries
                for key, value in project_data.items():
                    if key in config_data and isinstance(config_data[key], dict) and isinstance(value, dict):
                        config_data[key].update(value)
                    else:
                        config_data[key] = value

        return cls(
            wandb=WandbConfig(**config_data.get("wandb", {})),
            observatory=ObservatoryConfig(**config_data.get("observatory", {})),
            storage=StorageConfig(**config_data.get("storage", {})),
            datadog=DatadogConfig(**config_data.get("datadog", {})),
            ignore_env_vars=config_data.get("ignore_env_vars", False),
            profile=config_data.get("profile", "external"),
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
