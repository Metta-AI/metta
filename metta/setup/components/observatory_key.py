import os
import subprocess

from metta.app_backend.clients.base_client import get_machine_token
from metta.common.util.constants import DEV_STATS_SERVER_URI, PROD_STATS_SERVER_URI
from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.saved_settings import get_saved_settings
from metta.setup.utils import error, info, success, warning


@register_module
class ObservatoryKeySetup(SetupModule):
    install_once = True
    server_url: str = "https://observatory.softmax-research.net/api"
    extra_server_urls: list[str] = [PROD_STATS_SERVER_URI]

    @property
    def name(self) -> str:
        return "observatory-key"

    @property
    def description(self) -> str:
        return "Observatory auth key"

    def get_token(self, server_url: str | None = None) -> str | None:
        """Get token for specific server using the shared implementation"""
        return get_machine_token(server_url)

    def check_installed(self) -> bool:
        # Check if we have a token for this server
        server_urls = [self.server_url] + self.extra_server_urls
        for server_url in server_urls:
            if get_machine_token(server_url) is None:
                return False
        return True

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        info(f"Setting up Observatory authentication for {self.server_url}...")
        login_script = self.repo_root / "devops" / "observatory_login.py"

        if not login_script.exists():
            error("Observatory login script not found at devops/observatory_login.py")
            return

        try:
            # In test/CI environments or non-interactive mode, skip interactive OAuth to avoid opening browsers
            if os.environ.get("METTA_TEST_ENV") or os.environ.get("CI") or non_interactive:
                warning("Skipping Observatory interactive login in non-interactive/test/CI environment.")
            else:
                cmd = [str(login_script), self.server_url]
                if force:
                    cmd.append("--force")
                self.run_command(cmd, capture_output=False, non_interactive=non_interactive)
            success(f"Observatory auth configured for {self.server_url}")
        except subprocess.CalledProcessError:
            warning(f"""
            Observatory login failed. You can manually run:
                uv run python devops/observatory_login.py {self.server_url} [--force]
            """)

    def check_connected_as(self) -> str | None:
        # Check for token for this server
        token = get_machine_token(self.server_url)

        if token:
            # NOTE: We should do the api/whoami check once it is working
            return "@stem.ai"

        return None

    @property
    def can_remediate_connected_status_with_install(self) -> bool:
        return True

    def to_config_settings(self) -> dict[str, str | None]:
        if self.is_enabled() and get_saved_settings().user_type.is_softmax:
            return {"stats_server_uri": PROD_STATS_SERVER_URI}

        return {"stats_server_uri": None}


@register_module
class ObservatoryKeyLocalSetup(ObservatoryKeySetup):
    server_url: str = DEV_STATS_SERVER_URI
    extra_server_urls: list[str] = []

    @property
    def name(self) -> str:
        return "observatory-key-local"

    @property
    def description(self) -> str:
        return "Observatory auth key (local development)"
