import os
import re
import subprocess

from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
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

    def check_installed(self) -> bool:
        if os.environ.get("WANDB_API_KEY"):
            return True

        netrc_path = os.path.expanduser("~/.netrc")
        if os.path.exists(netrc_path):
            with open(netrc_path, "r") as f:
                return "api.wandb.ai" in f.read()

        return False

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        """Set up Weights & Biases authentication and configuration.

        Handles different user types:
        - SOFTMAX: Uses internal W&B setup
        - SOFTMAX_DOCKER: Expects W&B access via environment variables
        - Others: Provides guidance for manual setup

        Args:
            non_interactive: If True, skip interactive authentication prompts
        """
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
            return
        else:
            info("""
                To use Weights & Biases, you'll need an account.
                Visit https://wandb.ai/authorize to get your API key.
            """)

        # In test/CI environments or non-interactive mode, avoid interactive prompts entirely
        if os.environ.get("METTA_TEST_ENV") or os.environ.get("CI") or non_interactive:
            info("Skipping W&B interactive setup in non-interactive/test/CI environment.")
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
                match = re.search(r"Currently logged in as: (\S+) \(([^)]+)\)", output)
                if match:
                    return match.group(2)
            return None
        except Exception:
            return None

    @property
    def can_remediate_connected_status_with_install(self) -> bool:
        return True

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

                # TODO: let users specify their intended entity and project as part of configuration
                return dict(
                    enabled=True,
                    entity=wandb.Api().default_entity or "",
                    project="",
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
