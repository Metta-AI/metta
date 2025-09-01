"""
Configuration components - separated from setup logic.
"""

from metta.config.interface import ConfigurableComponent
from metta.setup.utils import info


class WandbConfigComponent(ConfigurableComponent):
    """Configuration for Weights & Biases."""

    @property
    def name(self) -> str:
        return "wandb"

    @property
    def description(self) -> str:
        return "Weights & Biases experiment tracking"

    def interactive_configure(self, current_config: dict) -> dict:
        """Interactive configuration for W&B."""
        info("\n📊 Configuring Weights & Biases...")
        info("Leave blank to keep current values")

        config = current_config.copy()

        # Show current values
        if config.get("entity"):
            info(f"Current entity: {config['entity']}")
        if config.get("project"):
            info(f"Current project: {config['project']}")

        # Get entity
        entity = input("W&B Entity/Team (blank to skip): ").strip()
        if entity:
            config["entity"] = entity

        # Get project
        project = input("W&B Project name (blank to skip): ").strip()
        if project:
            config["project"] = project

        # Ask if enabled
        current_enabled = config.get("enabled", True)
        enabled_str = "enabled" if current_enabled else "disabled"
        enabled = input(f"Enable W&B tracking? Currently {enabled_str} (y/n/blank to keep): ").strip().lower()
        if enabled == "y":
            config["enabled"] = True
        elif enabled == "n":
            config["enabled"] = False

        return config

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

    def interactive_configure(self, current_config: dict) -> dict:
        """Interactive configuration for storage."""
        info("\n Configuring Storage...")
        info("Leave blank to keep current values")

        config = current_config.copy()

        # Show current values
        if config.get("s3_bucket"):
            info(f"Current S3 bucket: {config['s3_bucket']}")
        if config.get("replay_dir"):
            info(f"Current replay directory: {config['replay_dir']}")

        # Get S3 bucket
        s3_bucket = input("S3 bucket name (e.g., my-company-metta): ").strip()
        if s3_bucket:
            config["s3_bucket"] = s3_bucket

            # Suggest S3 paths if bucket is provided
            use_s3 = input("Use S3 for replays? (y/n): ").strip().lower()
            if use_s3 == "y":
                config["replay_dir"] = f"s3://{s3_bucket}/replays/"

            use_s3_torch = input("Use S3 for torch profiler traces? (y/n): ").strip().lower()
            if use_s3_torch == "y":
                config["torch_profile_dir"] = f"s3://{s3_bucket}/torch_traces/"

            use_s3_checkpoints = input("Use S3 for checkpoints? (y/n): ").strip().lower()
            if use_s3_checkpoints == "y":
                config["checkpoint_dir"] = f"s3://{s3_bucket}/checkpoints/"

        # Get AWS profile
        aws_profile = input("AWS profile name (blank for default): ").strip()
        if aws_profile:
            config["aws_profile"] = aws_profile

        return config

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

    def interactive_configure(self, current_config: dict) -> dict:
        """Interactive configuration for Observatory."""
        info("\n Configuring Observatory...")
        info("Observatory is used for experiment tracking and remote evaluations")

        config = current_config.copy()

        # Show current values
        if config.get("stats_server_uri"):
            info(f"Current server: {config['stats_server_uri']}")

        # Ask if enabled
        current_enabled = config.get("enabled", False)
        enabled_str = "enabled" if current_enabled else "disabled"
        enabled = input(f"Enable Observatory? Currently {enabled_str} (y/n/blank to keep): ").strip().lower()
        if enabled == "y":
            config["enabled"] = True

            # Ask for custom server URI
            custom_uri = input("Stats server URI (blank for default): ").strip()
            if custom_uri:
                config["stats_server_uri"] = custom_uri
        elif enabled == "n":
            config["enabled"] = False

        return config

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

    def interactive_configure(self, current_config: dict) -> dict:
        """Interactive configuration for Datadog."""
        info("\n Configuring Datadog...")
        info("Datadog is used for monitoring and metrics")

        config = current_config.copy()

        # Ask if enabled
        current_enabled = config.get("enabled", False)
        enabled_str = "enabled" if current_enabled else "disabled"
        enabled = input(f"Enable Datadog? Currently {enabled_str} (y/n/blank to keep): ").strip().lower()
        if enabled == "y":
            config["enabled"] = True
            info("Note: Datadog API keys should be set as environment variables DD_API_KEY and DD_APP_KEY")
        elif enabled == "n":
            config["enabled"] = False

        return config

    def apply_defaults(self, config: dict, profile: str) -> dict:
        """Apply profile-based defaults."""
        if profile == "softmax":
            config.setdefault("enabled", True)
        else:
            config.setdefault("enabled", False)

        return config


# Registry of all configuration components
CONFIGURATION_COMPONENTS = {
    "wandb": WandbConfigComponent(),
    "storage": StorageConfigComponent(),
    "observatory": ObservatoryConfigComponent(),
    "datadog": DatadogConfigComponent(),
}
