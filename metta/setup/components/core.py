import os
import subprocess
import sys

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, success


@register_module
class CoreSetup(SetupModule):
    always_required = True

    def dependencies(self) -> list[str]:
        return ["system"]  # Changed: core depends on system (system installs bootstrap deps first)

    @property
    def description(self) -> str:
        return "Core Python dependencies and virtual environment"

    def check_installed(self) -> bool:
        # System deps (bazel, nimby, nim, git, g++) are now checked by system.py
        # Only check uv here (needed for this module to run)
        try:
            subprocess.run(["uv", "--version"], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            error("uv is not installed. Please install it using `./install.sh`")
            sys.exit(1)

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        cmd = ["uv", "sync"]
        cmd.extend(["--force-reinstall", "--no-cache"] if force else [])
        env = os.environ.copy()
        env["METTAGRID_FORCE_NIM_BUILD"] = "1"
        self.run_command(cmd, non_interactive=non_interactive, env=env, capture_output=False)
        success("Core dependencies installed")
