import os
import subprocess

from metta.app_backend.clients.base_client import NotAuthenticatedError, get_machine_token
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.util.constants import DEV_STATS_SERVER_URI, OBSERVATORY_AUTH_SERVER_URL, PROD_STATS_SERVER_URI
from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, info, success, warning


@register_module
class ObservatoryKeySetup(SetupModule):
    install_once = True
    auth_server_url: str = OBSERVATORY_AUTH_SERVER_URL
    api_server_url: str = PROD_STATS_SERVER_URI

    @property
    def name(self) -> str:
        return "observatory-key"

    @property
    def description(self) -> str:
        return "Observatory auth key"

    def check_installed(self) -> bool:
        # Check if we have a token for this server
        return get_machine_token(self.api_server_url) is not None

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        info(f"Setting up Observatory authentication for {self.api_server_url}...")
        login_script = self.repo_root / "devops" / "observatory_login.py"

        if not login_script.exists():
            error("Observatory login script not found at devops/observatory_login.py")
            return

        try:
            # In test/CI environments or non-interactive mode, skip interactive OAuth to avoid opening browsers
            if os.environ.get("METTA_TEST_ENV") or os.environ.get("CI") or non_interactive:
                warning("Skipping Observatory interactive login in non-interactive/test/CI environment.")
            else:
                cmd = [str(login_script), self.auth_server_url, self.api_server_url]
                if force:
                    cmd.append("--force")
                self.run_command(cmd, capture_output=False, non_interactive=non_interactive)
            success(f"Observatory auth configured for {self.api_server_url}")
        except subprocess.CalledProcessError:
            warning(f"""
            Observatory login failed. You can manually run:
                uv run python devops/observatory_login.py {self.auth_server_url} {self.api_server_url} [--force]
            """)

    def check_connected_as(self) -> str | None:
        try:
            StatsClient.create(self.api_server_url)
        except NotAuthenticatedError:
            return None
        return "@stem.ai"

    @property
    def can_remediate_connected_status_with_install(self) -> bool:
        return True

    def to_config_settings(self) -> dict[str, str | None]:
        if self.check_installed():
            return {"stats_server_uri": self.api_server_url}

        return {"stats_server_uri": None}


@register_module
class ObservatoryKeyLocalSetup(ObservatoryKeySetup):
    auth_server_url: str = OBSERVATORY_AUTH_SERVER_URL
    api_server_url: str = DEV_STATS_SERVER_URI

    @property
    def name(self) -> str:
        return "observatory-key-local"

    @property
    def description(self) -> str:
        return "Observatory auth key (local development)"
