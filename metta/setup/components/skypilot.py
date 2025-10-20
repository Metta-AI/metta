import os
import signal
import subprocess

from metta.common.util.constants import METTA_SKYPILOT_URL
from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.saved_settings import get_saved_settings
from metta.setup.utils import info, success


@register_module
class SkypilotSetup(SetupModule):
    install_once = True

    softmax_url = METTA_SKYPILOT_URL

    def dependencies(self) -> list[str]:
        return ["aws"]

    @property
    def description(self) -> str:
        return "SkyPilot cloud compute orchestration"

    def check_installed(self) -> bool:
        try:
            result = subprocess.run(["sky", "--version"], capture_output=True, text=True)
            return result.returncode == 0 and self._check_gh_auth()
        except FileNotFoundError:
            return False

    def _check_gh_auth(self) -> bool:
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        # TODO: check if the sdk version from outside of this uv environment matches the latest version.
        # It's possible that the user's `sky` is not the same as the one installed by the uv environment we are in.

        if not get_saved_settings().user_type.is_softmax:
            info("SkyPilot is only supported for Softmax users. Skipping...")
            return

        info("Setting up SkyPilot...")

        # In CI/test environments or non-interactive mode, avoid interactive login flows altogether
        if non_interactive:
            info("Skypilot installation requires interactive login. Skipping...")
            return

        # Check and setup GitHub authentication first
        # This is required because skypilot's launch.py uses 'gh pr list'
        if not self._check_gh_auth():
            info("GitHub CLI authentication required for SkyPilot...")
            info("SkyPilot uses 'gh' to check PR status when launching jobs.")
            # In non-interactive/test environments, skip attempting to open a browser
            if not (os.environ.get("METTA_TEST_ENV") or os.environ.get("CI") or non_interactive):
                try:
                    subprocess.run(["gh", "auth", "login", "--web"], check=False)
                except subprocess.CalledProcessError:
                    info("GitHub authentication may have been cancelled or failed.")
                    info("You can complete it later with: gh auth login")

        connected_as = self.check_connected_as()
        if connected_as == self.softmax_url and not force:
            info("""
            SkyPilot is already configured for a softmax user. Skipping authentication.
            You can force re-authentication with --force.
            """)
            return
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

    @property
    def can_remediate_connected_status_with_install(self) -> bool:
        return (
            # SkypilotSetup.install only implements authenticating with softmax
            get_saved_settings().user_type.is_softmax
            # If the connection is unhealthy, force installing will not help
            and self.check_connected_as() != f"{self.softmax_url} (unhealthy)"
        )

    def check_connected_as(self) -> str | None:
        if get_saved_settings().user_type.is_softmax:
            if not self._check_gh_auth():
                return None

            try:
                result = subprocess.run(["uv", "run", "--active", "sky", "api", "info"], capture_output=True, text=True)

                if result.returncode == 0:
                    if self.softmax_url in result.stdout:
                        if "healthy" in result.stdout.lower():
                            return f"{self.softmax_url}"
                        else:
                            return f"{self.softmax_url} (unhealthy)"
                return None
            except Exception:
                return None
        return None
