import os
import subprocess
import sys

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, success


@register_module
class UvSetup(SetupModule):
    always_required = True

    def dependencies(self) -> list[str]:
        return ["system"]

    @property
    def description(self) -> str:
        return "Python dependencies via uv sync"

    def check_installed(self) -> bool:
        try:
            subprocess.run(["uv", "--version"], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            error("uv is not installed. Please install it using `./install.sh`")
            sys.exit(1)

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        cmd = ["uv", "sync", "--all-packages"]
        cmd.extend(["--force-reinstall", "--no-cache"] if force else [])
        env = os.environ.copy()
        env["METTAGRID_FORCE_NIM_BUILD"] = "1"
        self.run_command(cmd, non_interactive=non_interactive, env=env, capture_output=False)
        success("Python dependencies installed")
