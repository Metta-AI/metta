import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, success


@register_module
class SkypilotSetup(SetupModule):
    install_once = True

    def dependencies(self) -> list[str]:
        return ["aws"]

    @property
    def description(self) -> str:
        return "SkyPilot cloud compute orchestration"

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("skypilot") and self.config.is_component_enabled("aws")

    def check_installed(self) -> bool:
        try:
            result = subprocess.run(["sky", "--version"], capture_output=True, text=True)
            return result.returncode == 0 and self._check_gh_auth()
        except FileNotFoundError:
            return False

    def _check_gh_auth(self) -> bool:
        try:
            result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @property
    def setup_script_location(self) -> str | None:
        if self.config.user_type.is_softmax:
            return "devops/skypilot/install.sh"
        return None

    def install(self) -> None:
        info("Setting up SkyPilot...")

        # Check and setup GitHub authentication first
        # This is required because skypilot's launch.py uses 'gh pr list'
        if not self._check_gh_auth():
            info("GitHub CLI authentication required for SkyPilot...")
            info("SkyPilot uses 'gh' to check PR status when launching jobs.")
            try:
                subprocess.run(["gh", "auth", "login", "--web"], check=False)
            except subprocess.CalledProcessError:
                info("GitHub authentication may have been cancelled or failed.")
                info("You can complete it later with: gh auth login")

        if self.config.user_type.is_softmax:
            super().install()
            success("SkyPilot installed")
        else:
            info("""
                To use SkyPilot with your own AWS account:
                  1. Ensure AWS credentials are configured
                  2. Authenticate with uv run sky api login
            """)

    def check_connected_as(self) -> str | None:
        if not self.check_installed():
            return None

        if self.config.user_type.is_softmax:
            try:
                result = subprocess.run(["sky", "api", "info"], capture_output=True, text=True)
                softmax_url = "skypilot-api.softmax-research.net"

                if result.returncode == 0:
                    if softmax_url in result.stdout:
                        if "healthy" in result.stdout.lower():
                            return f"{softmax_url}"
                        else:
                            return f"{softmax_url} (unhealthy)"
                return None
            except Exception:
                return None
        else:
            try:
                result = subprocess.run(["sky", "check"], capture_output=True, text=True)
                if result.returncode == 0:
                    return "configured"
                return None
            except Exception:
                return None
