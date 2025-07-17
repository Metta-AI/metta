import platform
import subprocess
import sys
from pathlib import Path

from typing_extensions import override

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, info, success, warning


@register_module
class SystemSetup(SetupModule):
    install_once = True

    @property
    @override
    def description(self) -> str:
        return "System dependencies (Homebrew packages, etc.)"

    @override
    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("system")

    @property
    def supported_for_platform(self) -> bool:
        return platform.system() == "Darwin" or self._find_brew_path() is not None

    @override
    def check_installed(self) -> bool:
        if not self.supported_for_platform:
            # NOTE: need to implement this at some point
            return True

        if not (brew_path := self._find_brew_path()):
            return False

        brewfile_path = self.repo_root / "devops" / "macos" / "Brewfile"
        if not brewfile_path.exists():
            return False

        try:
            result = self.run_command([brew_path, "bundle", "check", "--file", str(brewfile_path)], check=False)
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @override
    def install(self) -> None:
        info("Setting up system dependencies...")

        if self.supported_for_platform:
            if platform.system() == "Darwin" and not self._find_brew_path():
                self._install_homebrew()
            self._run_brew_bundle("Brewfile")
            success("System dependencies installed")
        else:
            # NOTE: need to implement this at some point
            info("""
                You will need to install brew or can manage package installation manually.
                See devops/macos/Brewfile for the full list of recommended packages.

                If you are on a mettabox, you can run `./devops/mettabox/setup_machine.sh`.
            """)

    def _install_homebrew(self) -> None:
        info("Installing Homebrew...")
        try:
            # Run the Homebrew installer directly with subprocess to preserve TTY
            # This allows it to prompt for sudo password interactively
            install_cmd = (
                '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            )
            subprocess.run(install_cmd, shell=True, check=True)
            success(
                "Homebrew installed successfully. Please source your shell to add it to your path "
                "and re-run metta installation."
            )
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            error(f"Error installing Homebrew: {e}")
            sys.exit(1)

    def _run_brew_bundle(self, brewfile_name: str, force: bool = False, no_fail: bool = False) -> None:
        brewfile_path = self.repo_root / "devops" / "macos" / brewfile_name
        if not brewfile_path.exists():
            warning(f"{brewfile_name} not found at {brewfile_path}")
            return

        info(f"Installing packages from {brewfile_name}...")
        command = ["brew", "bundle", "--file", str(brewfile_path)]

        if force:
            command.append("--force")
        if no_fail:
            command.append("--no-upgrade")

        try:
            # Run with output visible to user
            _ = self.run_command(command, capture_output=False)
        except subprocess.CalledProcessError:
            if not force and not no_fail:
                warning("""

                    Some packages are already installed but not managed by Homebrew.
                    This is common and usually not a problem. You have a few options:

                    1. Continue anyway (recommended) - The setup will skip already installed packages
                    2. Force Homebrew to manage them - Run with --brew-force flag
                    3. Skip conflicting packages - Run with --brew-no-fail flag

                    For now, we'll continue with the installation...
                """)

                # Retry with no-upgrade to skip already installed packages
                self._run_brew_bundle(brewfile_name, no_fail=True)

    def _find_brew_path(self) -> str | None:
        """Find the Homebrew executable path."""
        for path in ["/opt/homebrew/bin/brew", "/usr/local/bin/brew"]:
            if Path(path).exists():
                return path
        return None
