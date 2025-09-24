import functools
import platform
import subprocess
import sys

import yaml
from typing_extensions import override

from metta.common.util.collections import remove_falsey
from metta.common.util.fs import get_repo_root
from metta.setup.components.base import SetupModule
from metta.setup.components.system_packages.installers.base import PackageInstaller
from metta.setup.components.system_packages.installers.brew import BrewInstaller
from metta.setup.components.system_packages.types import SystemDepsConfig
from metta.setup.registry import register_module
from metta.setup.utils import error, info, success, warning


@functools.cache
def get_package_installer() -> PackageInstaller | None:
    for installer in [BrewInstaller()]:  # , AptInstaller()]:
        if installer.is_available():
            return installer
    return None


def get_system_deps_config() -> SystemDepsConfig:
    with open(get_repo_root() / "metta/setup/components/system_packages/deps.yaml", "r") as f:
        data = yaml.safe_load(f) or {}
    return SystemDepsConfig(**data)


@register_module
class SystemSetup(SetupModule):
    install_once = True

    @property
    @override
    def description(self) -> str:
        return "System dependencies"

    @property
    def _installer(self) -> PackageInstaller | None:
        return get_package_installer()

    @property
    def _config(self) -> SystemDepsConfig:
        return get_system_deps_config()

    @override
    def check_installed(self) -> bool:
        if not self._installer:
            return True
        return self._installer.check_installed(
            remove_falsey([getattr(p, self._installer.name) for p in self._config.packages.values()])
        )

    def _get_applicable_packages(self):
        if not self._installer:
            return []
        return remove_falsey([getattr(p, self._installer.name) for p in self._config.packages.values()])

    @override
    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        info("Setting up system dependencies...")

        if not self._installer:
            warning("""
                No supported package manager found.

                Supported package managers:
                - Homebrew (macOS/Linux)
                - APT (Debian/Ubuntu)

                Please install one of these package managers or install dependencies manually.
                See devops/system-deps.yaml for the full list of recommended packages.

                If you are on a mettabox, you can run `./devops/mettabox/setup_machine.sh`.
            """)
            return

        if not self._config:
            error("Failed to load system dependencies configuration")
            return

        # Install Homebrew on macOS if not present
        if platform.system() == "Darwin" and not isinstance(self._installer, BrewInstaller):
            self._install_homebrew()
            get_package_installer.cache_clear()

        if self._installer:
            self._installer.install(self._get_applicable_packages())

        success("System dependencies installed")

    def _install_homebrew(self) -> None:
        info("Installing Homebrew...")
        try:
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
