import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.profiles import UserType
from metta.setup.registry import register_module
from metta.setup.saved_settings import get_saved_settings
from metta.setup.utils import info, success, warning


@register_module
class CodeclipSetup(SetupModule):
    install_once = True  # Only install once, not every time

    @property
    def description(self) -> str:
        return "Developer tools copying code for LLM contexts"

    def _is_applicable(self) -> bool:
        # Only applicable for developer profiles, not docker profiles
        return get_saved_settings().user_type in [UserType.SOFTMAX, UserType.EXTERNAL, UserType.CLOUD]

    def check_installed(self) -> bool:
        """Check if codeclip is installed."""
        try:
            result = subprocess.run(["uv", "tool", "list"], capture_output=True, text=True, check=True)
            return "codeclip" in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install(self) -> None:
        """Install codeclip as an editable uv tool."""
        codeclip_dir = self.repo_root / "codebot" / "codeclip"
        gitlib_dir = self.repo_root / "gitlib"

        if not codeclip_dir.exists():
            warning(f"Codeclip directory not found at {codeclip_dir}")
            return

        info("Installing codeclip tool...")

        # Install as editable package using uv with gitlib dependency
        try:
            # First install the tool with gitlib
            self.run_command(
                ["uv", "tool", "install", "--force", "-e", str(codeclip_dir), "--with", f"gitlib @ file://{gitlib_dir}"]
            )

            success("Codeclip tool installed successfully!")
            info("You can now use 'metta clip' or 'codeclip' commands")
        except subprocess.CalledProcessError as e:
            warning(f"Failed to install codeclip: {e}")
            warning("You can manually install it with:")
            warning(f"  uv tool install --force -e {codeclip_dir} --with 'gitlib @ file://{gitlib_dir}'")
