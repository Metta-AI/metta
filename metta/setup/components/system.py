import functools
import platform
import subprocess
import sys

import typing_extensions
import yaml

import metta.common.util.collections
import metta.common.util.fs
import metta.setup.components.base
import metta.setup.components.system_packages.installers.base
import metta.setup.components.system_packages.installers.brew
import metta.setup.components.system_packages.types
import metta.setup.registry
import metta.setup.utils


@functools.cache
def get_package_installer() -> metta.setup.components.system_packages.installers.base.PackageInstaller | None:
    for installer in [metta.setup.components.system_packages.installers.brew.BrewInstaller()]:  # , AptInstaller()]:
        if installer.is_available():
            return installer
    return None


def get_system_deps_config() -> metta.setup.components.system_packages.types.SystemDepsConfig:
    with open(metta.common.util.fs.get_repo_root() / "metta/setup/components/system_packages/deps.yaml", "r") as f:
        data = yaml.safe_load(f) or {}
    return metta.setup.components.system_packages.types.SystemDepsConfig(**data)


@metta.setup.registry.register_module
class SystemSetup(metta.setup.components.base.SetupModule):
    install_once = True

    @property
    @typing_extensions.override
    def description(self) -> str:
        return "System dependencies"

    @property
    def _installer(self) -> metta.setup.components.system_packages.installers.base.PackageInstaller | None:
        return get_package_installer()

    @property
    def _config(self) -> metta.setup.components.system_packages.types.SystemDepsConfig:
        return get_system_deps_config()

    @typing_extensions.override
    def check_installed(self) -> bool:
        if not self._installer:
            return True
        return self._installer.check_installed(
            metta.common.util.collections.remove_falsey(
                [getattr(p, self._installer.name) for p in self._config.packages.values()]
            )
        )

    def _get_applicable_packages(self):
        if not self._installer:
            return []
        return metta.common.util.collections.remove_falsey(
            [getattr(p, self._installer.name) for p in self._config.packages.values()]
        )

    @typing_extensions.override
    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        metta.setup.utils.info("Setting up system dependencies...")

        if not self._installer:
            metta.setup.utils.warning("""
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
            metta.setup.utils.error("Failed to load system dependencies configuration")
            return

        # Install Homebrew on macOS if not present
        if platform.system() == "Darwin" and not isinstance(
            self._installer, metta.setup.components.system_packages.installers.brew.BrewInstaller
        ):
            self._install_homebrew()
            get_package_installer.cache_clear()

        if self._installer:
            self._installer.install(self._get_applicable_packages())

        metta.setup.utils.success("System dependencies installed")

    def _install_homebrew(self) -> None:
        metta.setup.utils.info("Installing Homebrew...")
        try:
            install_cmd = (
                '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            )
            subprocess.run(install_cmd, shell=True, check=True)
            metta.setup.utils.success(
                "Homebrew installed successfully. Please source your shell to add it to your path "
                "and re-run metta installation."
            )
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            metta.setup.utils.error(f"Error installing Homebrew: {e}")
            sys.exit(1)
