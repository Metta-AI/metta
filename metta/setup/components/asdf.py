import platform
import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.config import UserType
from metta.setup.registry import register_module
from metta.setup.utils import info, success, warning


@register_module
class AsdfSetup(SetupModule):
    @property
    def description(self) -> str:
        return "asdf version manager"

    def is_applicable(self) -> bool:
        return (
            platform.system() == "Darwin"
            or platform.system() == "Linux"
            and self.config.user_type == UserType.SOFTMAX
            and self.config.is_component_enabled("asdf")
        )

    def check_installed(self) -> bool:
        try:
            result = subprocess.run(["asdf", "version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def install(self) -> None:
        info("Setting up asdf...")

        try:
            info("Installing asdf via Homebrew...")
            # Use subprocess directly to preserve TTY for sudo prompts
            subprocess.run(["brew", "install", "asdf"], check=True)
            success("asdf installed via Homebrew!")

        except subprocess.CalledProcessError:
            warning("""
                asdf installation failed. You can install manually:
                1. Run: brew install asdf
                2. Re-run this setup wizard

                For now, we'll skip asdf setup and continue with other components.
            """)
