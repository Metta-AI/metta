import os
import subprocess

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
from metta.config.schema import get_config
from metta.setup.components.base import SetupModule
from metta.setup.profiles import UserType
from metta.setup.registry import register_module
from metta.setup.saved_settings import get_saved_settings
from metta.setup.utils import info, success, warning


@register_module
class WandbSetup(SetupModule):
    install_once = True

    @property
    def description(self) -> str:
        return "Weights & Biases experiment tracking"

    def should_install(self) -> bool:
        """Install W&B only if user has enabled it in configuration."""
        config = get_config()
        return config.wandb.enabled

    def check_installed(self) -> bool:
        if os.environ.get("WANDB_API_KEY"):
            return True

        netrc_path = os.path.expanduser("~/.netrc")
        if os.path.exists(netrc_path):
            with open(netrc_path, "r") as f:
                return "api.wandb.ai" in f.read()

        return False

    def install(self) -> None:
        info("Setting up Weights & Biases...")

        if self.check_installed():
            success("W&B already configured")
            return

        saved_settings = get_saved_settings()
        if saved_settings.user_type == UserType.SOFTMAX:
            info("""
                Your Weights & Biases access should have been provisioned.
                If you don't have access, contact your team lead.

                Visit https://wandb.ai/authorize to get your API key.
            """)
        elif saved_settings.user_type == UserType.SOFTMAX_DOCKER:
            info("Weights & Biases access should be provided via environment variables.")
            info("Skipping W&B setup.")
        else:
            info("""
                To use Weights & Biases, you'll need an account.
                Visit https://wandb.ai/authorize to get your API key.
            """)

        # In test/CI environments, avoid interactive prompts entirely
        if os.environ.get("METTA_TEST_ENV") or os.environ.get("CI"):
            info("Skipping W&B interactive setup in test/CI environment.")
            return

        use_wandb = input("\nDo you have your API key ready? (y/n): ").strip().lower()
        if use_wandb != "y":
            info("Skipping W&B setup. You can configure it later with 'wandb login'")
            return

        try:
            subprocess.run(["wandb", "login"], check=True)
            success("W&B configured successfully")
        except subprocess.CalledProcessError:
            warning("W&B login failed. You can run 'wandb login' manually later.")

    def check_connected_as(self) -> str | None:
        try:
            result = subprocess.run(["wandb", "login"], capture_output=True, text=True)
            # W&B outputs login status to stderr, not stdout
            output = result.stderr if result.stderr else result.stdout
            if result.returncode == 0 and "Currently logged in as:" in output:
                import re

                match = re.search(r"Currently logged in as: (\S+) \(([^)]+)\)", output)
                if match:
                    return match.group(2)
            return None
        except Exception:
            return None

    def get_configuration_schema(self) -> dict[str, tuple[type, str, str | None]]:
        """Define configuration schema for wandb."""
        return {
            "entity": (str, "W&B Entity (team/organization name)", None),
            "project": (str, "W&B Project name", None),
            "enabled": (bool, "Enable W&B tracking", "true"),
        }

    def interactive_configure(self) -> dict[str, str | bool] | None:
        """Interactive configuration for W&B."""
        from metta.setup.utils import info

        info("\nConfiguring Weights & Biases...")
        info("Leave blank to use defaults or skip")

        config = {}

        # Check if already logged in
        if self.check_installed():
            info("✓ W&B credentials found")

            # Try to get default entity
            try:
                import wandb

                default_entity = wandb.Api().default_entity
                if default_entity:
                    use_default = input(f"Use default entity '{default_entity}'? (y/n): ").lower()
                    if use_default == "y":
                        config["entity"] = default_entity
            except Exception:
                pass
        else:
            info("No W&B credentials found. Run 'wandb login' to authenticate.")

        # Get entity if not set
        if "entity" not in config:
            entity = input("W&B Entity/Team (leave blank for personal): ").strip()
            if entity:
                config["entity"] = entity

        # Get project
        project = input("W&B Project name (leave blank for default): ").strip()
        if project:
            config["project"] = project

        # Ask if enabled
        enabled = input("Enable W&B tracking? (y/n) [y]: ").strip().lower()
        config["enabled"] = enabled != "n"

        return config

    def export_env_vars(self, config: dict[str, str | bool]) -> dict[str, str]:
        """Export W&B configuration as environment variables."""
        env_vars = {}

        if config.get("enabled", True):
            env_vars["WANDB_ENABLED"] = "true"
            if config.get("entity"):
                env_vars["WANDB_ENTITY"] = config["entity"]
            if config.get("project"):
                env_vars["WANDB_PROJECT"] = config["project"]
        else:
            env_vars["WANDB_ENABLED"] = "false"

        return env_vars

    def to_config_settings(self) -> dict[str, str | bool]:
        saved_settings = get_saved_settings()
        if saved_settings.user_type.is_softmax:
            return dict(
                enabled=True,
                project=METTA_WANDB_PROJECT,
                entity=METTA_WANDB_ENTITY,
            )
        if self.is_enabled():
            try:
                import wandb

                # Check for configured values
                from metta.setup.config_manager import get_config_manager

                config = get_config_manager().get_component_config("wandb")

                return dict(
                    enabled=config.get("enabled", True),
                    entity=config.get("entity", wandb.Api().default_entity or ""),
                    project=config.get("project", ""),
                )
            except Exception:
                return dict(
                    enabled=True,
                    project="",
                    entity="",
                )
        return dict(
            enabled=False,
            project="",
            entity="",
        )
