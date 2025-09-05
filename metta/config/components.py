"""
Configuration components - separated from setup logic.
"""

from metta.config.interface import ConfigurableComponent


class WandbConfigComponent(ConfigurableComponent):
    """Configuration for Weights & Biases."""

    @property
    def name(self) -> str:
        return "wandb"

    @property
    def description(self) -> str:
        return "Weights & Biases experiment tracking"

    def apply_defaults(self, config: dict, profile: str) -> dict:
        """Apply profile-based defaults."""
        if profile == "softmax":
            # Softmax users get org defaults
            config.setdefault("enabled", True)
            config.setdefault("entity", "softmax-ai")
            config.setdefault("project", "metta")
        else:
            # External/cloud users start with disabled
            config.setdefault("enabled", False)

        return config


class StorageConfigComponent(ConfigurableComponent):
    """Configuration for storage (S3/local)."""

    @property
    def name(self) -> str:
        return "storage"

    @property
    def description(self) -> str:
        return "Storage configuration (S3 and local paths)"

    def apply_defaults(self, config: dict, profile: str) -> dict:
        """Apply profile-based defaults."""
        if profile == "softmax":
            # Softmax users get S3 defaults
            config.setdefault("replay_dir", "s3://softmax-public/replays/")
            config.setdefault("torch_profile_dir", "s3://softmax-public/torch_traces/")
            config.setdefault("checkpoint_dir", "./train_dir/checkpoints/")
        else:
            # External/cloud users get local defaults
            config.setdefault("replay_dir", "./train_dir/replays/")
            config.setdefault("torch_profile_dir", "./train_dir/torch_traces/")
            config.setdefault("checkpoint_dir", "./train_dir/checkpoints/")

        return config


class ObservatoryConfigComponent(ConfigurableComponent):
    """Configuration for Observatory."""

    @property
    def name(self) -> str:
        return "observatory"

    @property
    def description(self) -> str:
        return "Observatory experiment tracking and remote evaluations"

    def apply_defaults(self, config: dict, profile: str) -> dict:
        """Apply profile-based defaults."""
        if profile == "softmax":
            config.setdefault("enabled", True)
            config.setdefault("stats_server_uri", "https://observatory.softmax-research.net/api")
        else:
            config.setdefault("enabled", False)

        return config


class DatadogConfigComponent(ConfigurableComponent):
    """Configuration for Datadog."""

    @property
    def name(self) -> str:
        return "datadog"

    @property
    def description(self) -> str:
        return "Datadog monitoring and metrics"

    def apply_defaults(self, config: dict, profile: str) -> dict:
        """Apply profile-based defaults."""
        if profile == "softmax":
            config.setdefault("enabled", True)
        else:
            config.setdefault("enabled", False)

        return config


def get_configuration_components() -> dict:
    """
    Get configuration components with auto-discovery.

    Returns a mapping of component names to their ConfigurableComponent instances.
    Auto-discovers from SetupModules when available, with fallback to static registry.
    """
    components = {}

    # Static registry as fallback
    static_components = {
        "wandb": WandbConfigComponent,
        "storage": StorageConfigComponent,
        "observatory": ObservatoryConfigComponent,
        "datadog": DatadogConfigComponent,
    }

    # Try to auto-discover from SetupModules
    try:
        from metta.setup.registry import get_all_modules

        # Create a mapping of setup module names to config components
        module_to_component = {
            "wandb": WandbConfigComponent,
            "aws": StorageConfigComponent,  # AWSSetup handles storage config
            "observatory_key": ObservatoryConfigComponent,
            "datadog_agent": DatadogConfigComponent,
        }

        for module in get_all_modules():
            # Check if this module has a corresponding config component
            component_class = module_to_component.get(module.name)
            if component_class:
                # Use the module's name for consistency
                config_name = {"aws": "storage", "observatory_key": "observatory", "datadog_agent": "datadog"}.get(
                    module.name, module.name
                )

                if config_name not in components:
                    components[config_name] = component_class()

    except ImportError as e:
        # Fall back to static registry only if setup registry is not available
        import logging

        logging.warning(f"Failed to auto-discover config components: {e}. Using static registry.")

    # Ensure we always have the core components (fill in any missing ones)
    for name, component_class in static_components.items():
        if name not in components:
            components[name] = component_class()

    return components


# Registry of all configuration components
CONFIGURATION_COMPONENTS = get_configuration_components()
