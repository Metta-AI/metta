import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, success, warning


@register_module
class CodebotSetup(SetupModule):
    install_once = True

    @property
    def description(self) -> str:
        return "Codebot tools for AI development (includes codeclip)"

    def check_installed(self) -> bool:
        """Check if codebot is installed."""
        try:
            result = subprocess.run(["uv", "tool", "list"], capture_output=True, text=True, check=True)
            return "codebot" in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        """Install codebot as a uv tool."""
        codebot_dir = self.repo_root / "packages" / "codebot"

        if not codebot_dir.exists():
            warning(f"Codebot directory not found at {codebot_dir}")
            return

        info("Installing codebot tools...")
        cmd = ["uv", "tool", "install"]
        if force:
            cmd.append("--force")
        try:
            self.run_command([*cmd, "-e", str(codebot_dir)])
        except subprocess.CalledProcessError:
            # Fallback: try without editable flag
            try:
                self.run_command([*cmd, str(codebot_dir)])
            except subprocess.CalledProcessError as e:
                warning(f"""
                Failed to install codebot: {e}
                You can manually install it with:
                    uv tool install --force {codebot_dir}
                """)
                return

        success("Codebot tools installed successfully!")
        info("You can now use 'codeclip' command")
