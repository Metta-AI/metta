"""Configuration manager for component-based settings."""

from pathlib import Path
from typing import Any

import yaml

from metta.setup.registry import get_all_modules
from metta.setup.utils import error, info, success, warning


class ConfigManager:
    """Manages configuration for all components in a unified way."""

    def __init__(self):
        self.config_path = Path.home() / ".metta" / "config.yaml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from disk."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_config(self) -> None:
        """Save configuration to disk."""
        with open(self.config_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=True)

    def get_component_config(self, component: str) -> dict[str, Any]:
        """Get configuration for a specific component."""
        return self._config.get("components", {}).get(component, {})

    def set_component_config(self, component: str, key: str, value: Any) -> None:
        """Set a configuration value for a component."""
        if "components" not in self._config:
            self._config["components"] = {}
        if component not in self._config["components"]:
            self._config["components"][component] = {}

        self._config["components"][component][key] = value
        self._save_config()

    def get_all_config(self) -> dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()

    def export_env_vars(self) -> dict[str, str]:
        """Export all configuration as environment variables."""
        env_vars = {}

        # Get all modules and let them contribute env vars
        modules = get_all_modules()
        for module in modules.values():
            if hasattr(module, "export_env_vars"):
                component_config = self.get_component_config(module.name)
                env_vars.update(module.export_env_vars(component_config))

        return env_vars

    def interactive_configure(self, component_name: str | None = None) -> None:
        """Interactive configuration walkthrough."""
        modules = get_all_modules()

        if component_name:
            # Configure specific component
            if component_name not in modules:
                error(f"Unknown component: {component_name}")
                return

            module = modules[component_name]
            self._configure_component(module)
        else:
            # Walk through all configurable components
            info("Welcome to Metta configuration wizard!")
            info("I'll walk you through setting up each component.\n")

            for _name, module in modules.items():
                if hasattr(module, "get_configuration_schema"):
                    response = input(f"\nWould you like to configure {module.description}? (y/n): ")
                    if response.lower() == "y":
                        self._configure_component(module)

    def _configure_component(self, module) -> None:
        """Configure a single component interactively."""
        info(f"\nConfiguring {module.description}...")

        if hasattr(module, "interactive_configure"):
            # Component has custom configuration logic
            config = module.interactive_configure()
            if config:
                for key, value in config.items():
                    self.set_component_config(module.name, key, value)
                success(f"✓ {module.name} configured")
        elif hasattr(module, "get_configuration_schema"):
            # Component has configuration schema
            schema = module.get_configuration_schema()
            current_config = self.get_component_config(module.name)

            for key, (_field_type, description, default) in schema.items():
                current_value = current_config.get(key, default)

                # Show current value if set
                prompt = f"{description}"
                if current_value and current_value != default:
                    prompt += f" [current: {current_value}]"
                elif default:
                    prompt += f" [default: {default}]"
                prompt += ": "

                value = input(prompt).strip()

                if not value:
                    # Use current or default
                    value = current_value if current_value else default

                if value and value != default:
                    # Only save non-default values
                    self.set_component_config(module.name, key, value)

            success(f"✓ {module.name} configured")
        else:
            warning(f"{module.name} does not support configuration")


# Global instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
