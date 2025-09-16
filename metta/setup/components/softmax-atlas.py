import json
import subprocess
from pathlib import Path

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info
from softmax.asana.app_authenticate import login
from softmax.atlas.server import (
    SOFTMAX_ATLAS_ASANA_APP,
    SOFTMAX_ATLAS_NAME,
    get_atlas_asana_client,
)


@register_module
class SoftmaxAtlasSetup(SetupModule):
    install_once = True

    @property
    def name(self) -> str:
        return "softmax-atlas"

    @property
    def description(self) -> str:
        return f"MCP server supporting {SOFTMAX_ATLAS_NAME} in the Claude desktop app."

    def _check_asana_app_authenticated(self) -> bool:
        try:
            get_atlas_asana_client()
        except Exception:
            return False
        return True

    def check_installed(self) -> bool:
        return self._check_mcp_server_installed()  # and self._check_asana_app_authenticated()

    def _check_mcp_server_installed(self) -> bool:
        desktop_config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        if desktop_config_path.exists():
            with open(desktop_config_path, "r") as f:
                config = json.load(f)
                return SOFTMAX_ATLAS_NAME in config.get("mcpServers", {})
        # Assume it's elsewhere
        return True

    def _install_mcp_server(self) -> None:
        # Install server using editable install of the softmax package only
        subprocess.run(
            [
                "uv",
                "run",
                "mcp",
                "install",
                "--with-editable",
                str(self.repo_root / "softmax"),
                "./softmax/atlas/server.py",
            ],
            cwd=self.repo_root,
        )
        info(f"{SOFTMAX_ATLAS_NAME} installed successfully. You can use it within the Claude desktop app.")

    def _install_asana_app(self, non_interactive: bool = False, force: bool = False) -> None:
        if non_interactive:
            raise Exception("Non-interactive mode is not supported for Asana app authentication")
        login(SOFTMAX_ATLAS_ASANA_APP, force=force)
        info(f"{SOFTMAX_ATLAS_ASANA_APP} installed successfully. You can use it within the Claude desktop app.")

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        # if not self._check_asana_app_authenticated() or force:
        #     self._install_asana_app(non_interactive=non_interactive, force=force)
        print("installing mcp server")
        if force or not self._check_mcp_server_installed():
            print("doing installing")
            self._install_mcp_server()
