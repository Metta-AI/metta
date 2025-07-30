import subprocess

from metta.common.util.constants import DEV_STATS_SERVER_URI, PROD_STATS_SERVER_URI
from metta.common.util.stats_client_cfg import get_machine_token
from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, success, warning


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

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("observatory-key")

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

    def install(self) -> None:
        info(f"Setting up Observatory authentication for {self.server_url}...")
        login_script = self.repo_root / "devops" / "observatory_login.py"

        if login_script.exists():
            try:
                # Don't capture output - this is an interactive OAuth flow
                self.run_command([str(login_script), self.server_url], capture_output=False)
                success(f"Observatory auth configured for {self.server_url}")
            except subprocess.CalledProcessError:
                warning("Observatory login failed. You can manually run:")
                warning(f"  uv run python devops/observatory_login.py {self.server_url}")
        else:
            warning("Observatory login script not found at devops/observatory_login.py")

    def check_connected_as(self) -> str | None:
        # Check for token for this server
        token = get_machine_token(self.server_url)

        if token:
            # NOTE: We should do the api/whoami check once it is working
            return "@stem.ai"

        return None


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

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("observatory-key-local")
