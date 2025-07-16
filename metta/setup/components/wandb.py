import os
import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, success, warning


@register_module
class WandbSetup(SetupModule):
    install_once = True

    @property
    def description(self) -> str:
        return "Weights & Biases experiment tracking"

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("wandb")

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

        if self.config.user_type.is_softmax:
            info("""
                Your Weights & Biases access should have been provisioned.
                If you don't have access, contact your team lead.

                Visit https://wandb.ai/authorize to get your API key.
            """)
        else:
            info("""
                To use Weights & Biases, you'll need an account.
                Visit https://wandb.ai/authorize to get your API key.
            """)

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
