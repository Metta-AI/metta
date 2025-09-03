import subprocess
import sys

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, success


@register_module
class CoreSetup(SetupModule):
    @property
    def description(self) -> str:
        return "Core Python dependencies and virtual environment"

    def check_installed(self) -> bool:
        try:
            subprocess.run(["uv", "--version"], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            error("uv is not installed. Please install it first:")
            error("  curl -LsSf https://astral.sh/uv/install.sh | sh")
            sys.exit(1)

    def install(self, non_interactive: bool = False) -> None:
        self.run_command(["uv", "sync"], non_interactive=non_interactive)
        success("Core dependencies installed")
