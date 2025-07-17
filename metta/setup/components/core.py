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

    def is_applicable(self) -> bool:
        return True

    def check_installed(self) -> bool:
        try:
            subprocess.run(["uv", "--version"], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            error("uv is not installed. Please install it first:")
            error("  curl -LsSf https://astral.sh/uv/install.sh | sh")
            sys.exit(1)

    def install(self) -> None:
        self.run_command(["uv", "sync"])
        success("Core dependencies installed")

        # Verify imports after installation
        print("\nVerifying all local dependencies are importable...")
        deps_to_check = [
            "pufferlib",
            "metta.mettagrid.mettagrid_env",
            "metta.mettagrid.mettagrid_c",
        ]

        all_good = True
        for dep in deps_to_check:
            try:
                subprocess.run(
                    ["uv", "run", "python", "-c", f"import {dep}; print('Found {dep} at', {dep}.__file__)"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError:
                error(f"Failed to import {dep}")
                all_good = False

        if not all_good:
            error("Some dependencies failed to import. Please check your installation.")
