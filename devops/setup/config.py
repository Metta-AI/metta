from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml


class UserType(Enum):
    EXTERNAL = "external"
    CLOUD = "cloud"
    SOFTMAX = "softmax"
    SOFTMAX_DEVOPS = "softmax_devops"


PROFILE_DEFINITIONS = {
    UserType.EXTERNAL: {
        "components": {
            "system": {"enabled": True},
            "core": {"enabled": True},
            "githooks": {"enabled": True},
            "mettascope": {"enabled": True},
            "observatory-fe": {"enabled": False},
            "observatory-cli": {"enabled": False},
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
            "observatory-fe": {"enabled": True},
            "observatory-cli": {"enabled": False},
            "aws": {"enabled": True},
            "wandb": {"enabled": True},
            "skypilot": {"enabled": True},
            "tailscale": {"enabled": False},
        }
    },
    UserType.SOFTMAX: {
        "components": {
            "system": {"enabled": True},
            "core": {"enabled": True},
            "githooks": {"enabled": True},
            "mettascope": {"enabled": True},
            "observatory-fe": {"enabled": True},
            "observatory-cli": {"enabled": True, "expected_connection": "@stem.ai"},
            "aws": {"enabled": True, "expected_connection": "751442549699"},
            "wandb": {"enabled": True, "expected_connection": "metta-research"},
            "skypilot": {"enabled": True, "expected_connection": "skypilot-api.softmax-research.net"},
            "tailscale": {"enabled": True, "expected_connection": "@stem.ai"},
        }
    },
}
PROFILE_DEFINITIONS[UserType.SOFTMAX_DEVOPS] = PROFILE_DEFINITIONS[UserType.SOFTMAX].copy()


class SetupConfig:
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            self.config_path = Path.home() / ".metta" / "config.yaml"
        else:
            self.config_path = config_path

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config = self._load_config()

    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def save(self) -> None:
        with open(self.config_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
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

    def is_component_enabled(self, component: str) -> bool:
        comp_config = self.get(f"components.{component}", {})
        return comp_config.get("enabled", False) if isinstance(comp_config, dict) else bool(comp_config)

    def get_expected_connection(self, component: str) -> str | None:
        return self.get(f"components.{component}", {}).get("expected_connection")

    def apply_profile(self, profile: UserType) -> None:
        self.user_type = profile

        profile_config = PROFILE_DEFINITIONS.get(profile, {})
        for component, settings in profile_config.get("components", {}).items():
            for key, value in settings.items():
                self.set(f"components.{component}.{key}", value)
