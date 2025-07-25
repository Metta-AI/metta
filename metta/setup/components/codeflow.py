import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.config import UserType
from metta.setup.registry import register_module
from metta.setup.utils import info, success, warning


@register_module
class CodeflowSetup(SetupModule):
    install_once = True  # Only install once, not every time

    @property
    def description(self) -> str:
        return "Developer tools copying code for LLM contexts"

    def is_applicable(self) -> bool:
        # Only applicable for developer profiles, not docker profiles
        return self.config.user_type in [UserType.SOFTMAX, UserType.EXTERNAL, UserType.CLOUD]

    def check_installed(self) -> bool:
        """Check if codeflow is installed."""
        try:
            result = subprocess.run(["uv", "tool", "list"], capture_output=True, text=True, check=True)
            return "codeflow" in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install(self) -> None:
        """Install codeflow as an editable uv tool."""
        codeflow_dir = self.repo_root / "metta" / "setup" / "tools" / "codeflow"

        if not codeflow_dir.exists():
            warning(f"Codeflow directory not found at {codeflow_dir}")
            return

        info("Installing codeflow tool...")

        # Install as editable package using uv
        try:
            # Use --force to update if already installed
            self.run_command(["uv", "tool", "install", "--force", "-e", str(codeflow_dir)])
            success("Codeflow tool installed successfully!")
            info("You can now use 'metta code' or 'codeflow' commands")
        except subprocess.CalledProcessError as e:
            warning(f"Failed to install codeflow: {e}")
            warning("You can manually install it with:")
            warning(f"  cd {codeflow_dir} && uv tool install --force -e .")
