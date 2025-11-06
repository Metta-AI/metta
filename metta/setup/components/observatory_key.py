import os
import subprocess

import metta.app_backend.clients.base_client
import metta.common.util.constants
import metta.setup.components.base
import metta.setup.registry
import metta.setup.utils


@metta.setup.registry.register_module
class ObservatoryKeySetup(metta.setup.components.base.SetupModule):
    install_once = True
    auth_server_url: str = metta.common.util.constants.OBSERVATORY_AUTH_SERVER_URL
    api_server_url: str = metta.common.util.constants.PROD_STATS_SERVER_URI

    @property
    def name(self) -> str:
        return "observatory-key"

    @property
    def description(self) -> str:
        return "Observatory auth key"

    def check_installed(self) -> bool:
        # Check if we have a token for this server
        return metta.app_backend.clients.base_client.get_machine_token(self.api_server_url) is not None

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        metta.setup.utils.info(f"Setting up Observatory authentication for {self.api_server_url}...")
        login_script = self.repo_root / "devops" / "observatory_login.py"

        if not login_script.exists():
            metta.setup.utils.error("Observatory login script not found at devops/observatory_login.py")
            return

        try:
            # In test/CI environments or non-interactive mode, skip interactive OAuth to avoid opening browsers
            if os.environ.get("METTA_TEST_ENV") or os.environ.get("CI") or non_interactive:
                metta.setup.utils.warning(
                    "Skipping Observatory interactive login in non-interactive/test/CI environment."
                )
            else:
                cmd = [str(login_script), self.auth_server_url, self.api_server_url]
                if force:
                    cmd.append("--force")
                self.run_command(cmd, capture_output=False, non_interactive=non_interactive)
            metta.setup.utils.success(f"Observatory auth configured for {self.api_server_url}")
        except subprocess.CalledProcessError:
            metta.setup.utils.warning(f"""
            Observatory login failed. You can manually run:
                uv run python devops/observatory_login.py {self.auth_server_url} {self.api_server_url} [--force]
            """)

    def check_connected_as(self) -> str | None:
        # Check for token for this server
        token = metta.app_backend.clients.base_client.get_machine_token(self.api_server_url)

        if token:
            # NOTE: We should do the api/whoami check once it is working
            return "@stem.ai"

        return None

    @property
    def can_remediate_connected_status_with_install(self) -> bool:
        return True

    def to_config_settings(self) -> dict[str, str | None]:
        if self.check_installed():
            return {"stats_server_uri": self.api_server_url}

        return {"stats_server_uri": None}


@metta.setup.registry.register_module
class ObservatoryKeyLocalSetup(ObservatoryKeySetup):
    auth_server_url: str = metta.common.util.constants.DEV_STATS_SERVER_URI
    api_server_url: str = metta.common.util.constants.DEV_STATS_SERVER_URI

    @property
    def name(self) -> str:
        return "observatory-key-local"

    @property
    def description(self) -> str:
        return "Observatory auth key (local development)"
