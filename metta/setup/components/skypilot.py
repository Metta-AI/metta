import signal
import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, success


@register_module
class SkypilotSetup(SetupModule):
    install_once = False

    softmax_url = "skypilot-api.softmax-research.net"

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
            result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, timeout=2)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

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

        connected_as = self.check_connected_as()
        if connected_as == self.softmax_url:
            info("SkyPilot is already configured for a softmax user. Skipping authentication.")
        elif connected_as == "configured":
            info("SkyPilot is already configured for external use. Skipping authentication.")
        else:
            # Need to authenticate skypilot.
            if self.config.user_type.is_softmax:
                try:
                    # Temporarily block Ctrl+C for parent process during script execution
                    # This is necessary because `sky api login` flow requires ctrl+c before the token can be pasted.

                    # Note: it's important to pass lambda, not `signal.SIG_IGN`, otherwise ctrl+c would be blocked even
                    # in the child process.
                    original_sigint_handler = signal.signal(signal.SIGINT, lambda signum, frame: None)

                    self.run_command(["bash", "./devops/skypilot/install.sh"], capture_output=False)
                    success("SkyPilot installed")
                finally:
                    signal.signal(signal.SIGINT, original_sigint_handler)
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
                result = subprocess.run(["uv", "run", "sky", "api", "info"], capture_output=True, text=True)

                if result.returncode == 0:
                    if self.softmax_url in result.stdout:
                        if "healthy" in result.stdout.lower():
                            return f"{self.softmax_url}"
                        else:
                            return f"{self.softmax_url} (unhealthy)"
                return None
            except Exception:
                return None
        else:
            try:
                result = subprocess.run(["uv", "run", "sky", "check"], capture_output=True, text=True)
                if result.returncode == 0:
                    return "configured"
                return None
            except Exception:
                return None
