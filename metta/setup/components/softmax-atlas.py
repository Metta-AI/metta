import json
import os
import subprocess

from metta.common.util.fs import get_repo_root
from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info


@register_module
class SoftmaxAtlasSetup(SetupModule):
    install_once = True

    server_loc = get_repo_root() / "softmax" / "atlas.py"

    @property
    def name(self) -> str:
        return "softmax-atlas"

    @property
    def description(self) -> str:
        return "Softmax Atlas (texture atlas)"

    def check_installed(self) -> bool:
        if os.path.exists("~/Library/Application\\ Support/Claude/claude_desktop_config.json"):
            with open("~/Library/Application\\ Support/Claude/claude_desktop_config.json", "r") as f:
                config = json.load(f)
                return "Softmax Atlas" in config.get("mcpServers", {})
        # Assume it's elsewhere
        return True

    def install(self, non_interactive: bool = False) -> None:
        subprocess.run(["uv", "run", "mcp", "install", str(self.server_loc)])
        info("Softmax Atlas installed successfully. You can use it within the Claude desktop app.")
