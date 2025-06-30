import json
import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.config import UserType
from metta.setup.registry import register_module
from metta.setup.utils import error, info, success, warning


@register_module
class TailscaleSetup(SetupModule):
    @property
    def description(self) -> str:
        return "Tailscale VPN for internal network access"

    def is_applicable(self) -> bool:
        return self.config.user_type == UserType.SOFTMAX and self.config.is_component_enabled("tailscale")

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
                    2. Authenticate with Google using your @stem.ai account
                    3. Grant system extension permissions if prompted

                    See: https://tailscale.com/kb/1340/macos-sysext
                """)
            else:
                success(f"Tailscale connected as {current}")
            return

        info("""
            Your Tailscale access should have been provisioned.
            If you don't have access, contact your team lead.

            Installing Tailscale...

            Note: Tailscale installation requires sudo privileges.
            You may be prompted for your password.
        """)

        try:
            self.run_command(["brew", "install", "--cask", "tailscale"])
            success("Tailscale installed")

            info("""
                To complete setup:
                1. Look for the Tailscale application in your Applications folder
                2. Run: tailscale up
                3. Log in with Google using your @stem.ai account
                4. Grant system extension permissions when prompted

                For more info: https://tailscale.com/kb/1340/macos-sysext
            """)

        except subprocess.CalledProcessError as e:
            error(f"Failed to install Tailscale: {e}")
