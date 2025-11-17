import os
import shutil
import subprocess
import sys

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, success


@register_module
class CoreSetup(SetupModule):
    always_required = True

    @property
    def description(self) -> str:
        return "Core Python dependencies and virtual environment"

    def check_installed(self) -> bool:
        # TODO: cooling: remove partial redundancy with install.sh system dep existence checks
        # TODO: check versions
        # TODO: move some of this logic into components/system.py, and ideally have components/system.py
        # and have core.py and system.py checks run before requiring a full uv sync
        for system_dep in ["uv", "bazel", "git", "g++", "nimby", "nim"]:
            if not shutil.which(system_dep):
                error(f"{system_dep} is not installed. Please install it using `./install.sh`")
                sys.exit(1)
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
