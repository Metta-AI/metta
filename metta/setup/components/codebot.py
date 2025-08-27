import subprocess
from pathlib import Path

from metta.setup.components.base import SetupModule
from metta.setup.profiles import UserType
from metta.setup.registry import register_module
from metta.setup.saved_settings import get_saved_settings
from metta.setup.utils import info, success, warning


@register_module
class CodebotSetup(SetupModule):
    install_once = True  # Only install once, not every time

    @property
    def description(self) -> str:
        return "Codebot tools for AI development (includes codeclip)"

    def _is_applicable(self) -> bool:
        # Only applicable for developer profiles, not docker profiles
        return get_saved_settings().user_type in [UserType.SOFTMAX, UserType.EXTERNAL, UserType.CLOUD]

    def check_installed(self) -> bool:
        """Check if codebot is installed."""
        try:
            result = subprocess.run(["uv", "tool", "list"], capture_output=True, text=True, check=True)
            return "codebot" in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install(self) -> None:
        """Install codebot as an editable uv tool."""
        codebot_dir = self.repo_root / "codebot"
        gitta_dir = self.repo_root / "gitta"

        if not codebot_dir.exists():
            warning(f"Codebot directory not found at {codebot_dir}")
            return

        info("Installing codebot tools...")

        # Install codebot package which includes codeclip command
        try:
            # First install the tool with gitta dependency
            result = subprocess.run(
                ["uv", "tool", "install", "--force", "-e", str(codebot_dir), "--with", f"gitta @ file://{gitta_dir}"],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
            )

            if result.returncode != 0:
                # Try a different approach if editable install fails
                # Install without -e flag
                self.run_command(
                    ["uv", "tool", "install", "--force", str(codebot_dir), "--with", f"gitta @ file://{gitta_dir}"]
                )
        except subprocess.CalledProcessError:
            # Fallback: install as a regular package
            try:
                self.run_command(
                    ["uv", "tool", "install", "--force", str(codebot_dir), "--with", f"gitta @ file://{gitta_dir}"]
                )
            except subprocess.CalledProcessError as e:
                warning(f"Failed to install codebot: {e}")
                warning("You can manually install it with:")
                warning(f"  uv tool install --force {codebot_dir} --with 'gitta @ file://{gitta_dir}'")
                return

        # Fix the Python path for the install
        # uv doesn't set up the path correctly for our package structure
        uv_tool_path = Path.home() / ".local" / "share" / "uv" / "tools" / "codebot"
        if uv_tool_path.exists():
            # Find the site-packages directory
            for p in uv_tool_path.glob("lib/python*/site-packages"):
                pth_file = p / "__editable__.codebot-0.1.0.pth"
                pth_file.write_text(str(self.repo_root))
                info(f"Fixed Python path in {pth_file}")
                break

        success("Codebot tools installed successfully!")
        info("You can now use 'codeclip' command")
