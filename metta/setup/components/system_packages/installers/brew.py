import subprocess

from metta.setup.components.system_packages.installers.base import PackageInstaller
from metta.setup.components.system_packages.types import BrewPackageConfig
from metta.setup.utils import info


class BrewInstaller(PackageInstaller[BrewPackageConfig]):
    @property
    def name(self) -> str:
        return "brew"

    def is_available(self) -> bool:
        result = subprocess.run(["which", "brew"], check=False, capture_output=True)
        return result.returncode == 0

    def _get_list_cmd(self, cmd: list[str]) -> list[str]:
        return subprocess.run(cmd, text=True, check=True, capture_output=True).stdout.strip().split("\n")

    def _install_cmd(self, cmd: list[str]) -> None:
        subprocess.run(cmd, text=True, check=True, capture_output=False)

    def get_installed_casks(self) -> list[str]:
        return self._get_list_cmd(["brew", "list", "--cask"])

    def install_casks(self, casks: list[str]) -> None:
        installed = self._get_list_cmd(["brew", "list", "--cask"])
        if to_install := set(casks) - set(installed):
            info(f"Installing {', '.join(to_install)}...")
            self._install_cmd(["brew", "install", "--cask", *to_install])

    def _get_installed_state(self) -> tuple[list[str], list[str], list[str]]:
        installed = self._get_list_cmd(["brew", "list", "--formula"])
        pinned = self._get_list_cmd(["brew", "list", "--pinned"])
        tapped = self._get_list_cmd(["brew", "tap"])
        return installed, pinned, tapped

    def _get_changes_to_apply(self, packages: list[BrewPackageConfig]) -> tuple[list[str], list[str], list[str]]:
        installed, pinned, tapped = self._get_installed_state()
        to_install = []
        for package in packages:
            if not any(p in installed for p in filter(None, [package.name, package.installed_name])):
                to_install.append(package)
        to_pin = list(set(pinned) - set([p.fully_specified_name for p in packages if p.pin]))
        to_tap = list(set(tapped) - set([p.tap for p in packages if p.tap]))
        return to_install, to_pin, to_tap

    def check_installed(self, packages: list[BrewPackageConfig]) -> bool:
        return any(self._get_changes_to_apply(packages))

    def install(self, packages: list[BrewPackageConfig]) -> None:
        to_install, to_pin, to_tap = self._get_changes_to_apply(packages)
        if to_tap:
            info(f"Adding taps: {', '.join(to_tap)}")
            self._install_cmd(["brew", "tap", *to_tap])

        if to_install:
            full_install_names = [p.fully_specified_name for p in packages if p.name in to_install]
            info(f"Installing {full_install_names}...")
            self._install_cmd(["brew", "install", *full_install_names])

        if to_pin:
            info(f"Pinning {to_pin}...")
            self._install_cmd(["brew", "pin", *to_pin])
