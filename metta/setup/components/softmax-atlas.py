import json
import subprocess
from pathlib import Path

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info
from softmax.atlas import SOFTMAX_ATLAS_NAME, SOFTMAX_ATLAS_SERVER_PATH


@register_module
class SoftmaxAtlasSetup(SetupModule):
    install_once = True

    @property
    def name(self) -> str:
        return "softmax-atlas"

    @property
    def description(self) -> str:
        return f"MCP server supporting {SOFTMAX_ATLAS_NAME} in the Claude desktop app."

    def check_installed(self) -> bool:
        desktop_config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        if desktop_config_path.exists():
            with open(desktop_config_path, "r") as f:
                config = json.load(f)
                return SOFTMAX_ATLAS_NAME in config.get("mcpServers", {})
        # Assume it's elsewhere
        return True

    def install(self, non_interactive: bool = False) -> None:
        subprocess.run(["uv", "run", "mcp", "install", SOFTMAX_ATLAS_SERVER_PATH])
        info(f"{SOFTMAX_ATLAS_NAME} installed successfully. You can use it within the Claude desktop app.")
