from enum import Enum
from pathlib import Path
from typing import TypeVar, cast

import yaml
from typing_extensions import NotRequired, TypedDict


class UserType(Enum):
    EXTERNAL = "external"
    CLOUD = "cloud"
    SOFTMAX = "softmax"
    SOFTMAX_DOCKER = "softmax-docker"
    CUSTOM = "custom"

    @property
    def is_softmax(self) -> bool:
        return self in (UserType.SOFTMAX, UserType.SOFTMAX_DOCKER)

    def get_description(self) -> str:
        descriptions = {
            UserType.EXTERNAL: "External contributor",
            UserType.CLOUD: "User with own cloud account",
            UserType.SOFTMAX: "Softmax employee",
            UserType.SOFTMAX_DOCKER: "Softmax (Docker)",
            UserType.CUSTOM: "Custom configuration",
        }
        return descriptions.get(self, self.value)


class ComponentConfig(TypedDict):
    enabled: bool
    expected_connection: NotRequired[str | None]


class ProfileConfig(TypedDict):
    components: dict[str, ComponentConfig]


CURRENT_CONFIG_VERSION = 1


PROFILE_DEFINITIONS: dict[UserType, ProfileConfig] = {
    UserType.EXTERNAL: {
        "components": {
            "system": {"enabled": True},
            "core": {"enabled": True},
            "githooks": {"enabled": True},
            "mettascope": {"enabled": True},
            "observatory-key": {"enabled": False},
            "aws": {"enabled": False},
            "wandb": {"enabled": False},
            "skypilot": {"enabled": False},
            "tailscale": {"enabled": False},
        }
    },
    UserType.CLOUD: {
        "components": {
            "system": {"enabled": True},
            "core": {"enabled": True},
            "githooks": {"enabled": True},
            "mettascope": {"enabled": True},
            "observatory-key": {"enabled": False},
            "aws": {"enabled": True},
            "wandb": {"enabled": True},
            "skypilot": {"enabled": True},
            "tailscale": {"enabled": False},
        }
    },
    UserType.SOFTMAX_DOCKER: {
        "components": {
            "system": {"enabled": True},
            "core": {"enabled": True},
            "githooks": {"enabled": True},
            "mettascope": {"enabled": False},
            "observatory-key": {"enabled": False},
            "aws": {"enabled": True, "expected_connection": "751442549699"},
            "wandb": {"enabled": True, "expected_connection": "metta-research"},
            "skypilot": {"enabled": False},
            "tailscale": {"enabled": False},
        }
    },
    UserType.SOFTMAX: {
        "components": {
            "system": {"enabled": True},
            "core": {"enabled": True},
            "codeclip": {"enabled": True},
            "githooks": {"enabled": True},
            "mettascope": {"enabled": True},
            "observatory-key": {"enabled": True, "expected_connection": "@stem.ai"},
            "aws": {"enabled": True, "expected_connection": "751442549699"},
            "wandb": {"enabled": True, "expected_connection": "metta-research"},
            "skypilot": {"enabled": True, "expected_connection": "skypilot-api.softmax-research.net"},
            "tailscale": {"enabled": True, "expected_connection": "@stem.ai"},
        }
    },
}

T = TypeVar("T")


class SetupConfig:
    def __init__(self, config_path: Path | None = None):
        self.config_path: Path = config_path or Path.home() / ".metta" / "config.yaml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config: dict = self._load_config()

    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def save(self) -> None:
        with open(self.config_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def get(self, key: str, default: T) -> T:
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return cast(T, value)

    def set(self, key: str, value: T) -> None:
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        self.save()

    @property
    def user_type(self) -> UserType:
        return UserType(self.get("user_type", UserType.EXTERNAL.value))

    @user_type.setter
    def user_type(self, value: UserType) -> None:
        self.set("user_type", value.value)

    @property
    def config_version(self) -> int:
        return self.get("config_version", 0)

    @property
    def is_custom_config(self) -> bool:
        return self.get("custom_config", False)

    def get_components(self) -> dict[str, ComponentConfig]:
        """Get the components configuration based on mode."""
        if not self.is_custom_config:
            # Use dynamic profile resolution
            profile_config = PROFILE_DEFINITIONS.get(self.user_type, {})
            return profile_config.get("components", {})
        else:
            # Use saved configuration
            return self.get("components", {})

    def is_component_enabled(self, component: str) -> bool:
        components = self.get_components()
        comp_settings = components.get(component, {})
        return comp_settings.get("enabled", False)

    def get_expected_connection(self, component: str) -> str | None:
        components = self.get_components()
        comp_settings = components.get(component, {})
        return comp_settings.get("expected_connection")

    def apply_profile(self, profile: UserType) -> None:
        """Apply a profile configuration (uses dynamic resolution)."""
        self.user_type = profile
        self.set("custom_config", False)
        self.set("config_version", CURRENT_CONFIG_VERSION)
        # Remove any saved components to ensure dynamic resolution
        if "components" in self._config:
            del self._config["components"]
            self.save()

    def setup_custom_profile(self, base_profile: UserType) -> None:
        """Set up a custom configuration based on a profile."""
        self.user_type = base_profile
        self.set("custom_config", True)
        self.set("config_version", CURRENT_CONFIG_VERSION)

        # Copy all component settings from the base profile
        profile_config = PROFILE_DEFINITIONS.get(base_profile, {})
        for component, settings in profile_config.get("components", {}).items():
            for key, value in settings.items():
                self.set(f"components.{component}.{key}", value)
