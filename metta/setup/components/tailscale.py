import json
import platform
import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.config import UserType
from metta.setup.registry import register_module
from metta.setup.utils import info, success, warning


@register_module
class TailscaleSetup(SetupModule):
    install_once = True

    @property
    def description(self) -> str:
        return "Tailscale VPN for internal network access"

    def is_applicable(self) -> bool:
        return (
            platform.system() == "Darwin"
            and self.config.user_type == UserType.SOFTMAX
            and self.config.is_component_enabled("tailscale")
        )

    def check_installed(self) -> bool:
        try:
            result = subprocess.run(["tailscale", "version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def check_connected_as(self) -> str | None:
        if not self.check_installed():
            return None

        try:
            result = subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True)
            if result.returncode != 0:
                return None
            status = json.loads(result.stdout)
            if status.get("BackendState") != "Running":
                return None
            if "Self" in status and "UserID" in status["Self"]:
                user_id = status["Self"]["UserID"]
                if "User" in status and str(user_id) in status["User"]:
                    user_info = status["User"][str(user_id)]
                    login_name = user_info.get("LoginName", "")
                    if "@" in login_name:
                        return "@" + login_name.split("@")[1]
                    return login_name
                return "connected"

            return None

        except Exception:
            return None

    def install(self) -> None:
        info("Setting up Tailscale...")

        if self.check_installed():
            success("Tailscale already installed")

            # Check if running
            current = self.check_connected_as()
            if not current:
                warning("Tailscale is installed but not running")
                info("""
                    Your Tailscale access should have been provisioned.
                    If you don't have access, contact your team lead.

                    To connect:
                    1. Run: tailscale up
                    2. Authenticate with Google using your @stem.ai account if prompted
                    3. Grant system extension permissions if prompted

                    See: https://tailscale.com/kb/1340/macos-sysext
                """)
            else:
                success(f"Tailscale connected as {current}")
            return

        info("""
            Your Tailscale access should have been provisioned.
            If you don't have access, contact your team lead.
        """)

        try:
            info("Installing Tailscale via Homebrew...")
            # Use subprocess directly to preserve TTY for sudo prompts
            subprocess.run(["brew", "install", "--cask", "tailscale"], check=True)
            success("Tailscale installed via Homebrew!")

            warning("""
                IMPORTANT: Now you need to launch Tailscale manually

                Please do the following:
                1. Open Tailscale from Applications folder
                2. Grant system extension permissions if prompted
                3. Authenticate with Google using your @stem.ai account if prompted

                For more info: https://tailscale.com/kb/1340/macos-sysext
            """)

        except subprocess.CalledProcessError:
            warning("""
                Tailscale installation failed. You can install manually:
                1. Run: brew install --cask tailscale
                2. Open Tailscale from Applications folder
                3. Grant permissions in System Settings → Privacy & Security
                4. Re-run this setup wizard

                For now, we'll skip Tailscale setup and continue with other components.
            """)
