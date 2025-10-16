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

    def _get_changes_to_apply(
        self, packages: list[BrewPackageConfig]
    ) -> tuple[list[BrewPackageConfig], list[BrewPackageConfig], list[str]]:
        installed, pinned, tapped = self._get_installed_state()
        to_install: list[BrewPackageConfig] = []

        def _package_in_output_list(p: BrewPackageConfig, output_list: list[str]) -> bool:
            return any(p in output_list for p in filter(None, [p.name, p.installed_name]))

        to_install = [p for p in packages if not _package_in_output_list(p, installed)]
        to_pin = [p for p in packages if p.pin and not _package_in_output_list(p, pinned)]
        to_tap = list(set([p.tap for p in packages if p.tap and p.tap not in tapped]))
        return to_install, to_pin, to_tap

    def check_installed(self, packages: list[BrewPackageConfig]) -> bool:
        """Returns True when no changes are required."""
        to_install, to_pin, to_tap = self._get_changes_to_apply(packages)
        all_installed = not any([to_install, to_pin, to_tap])
        for label, packages in [("To install", to_install), ("To pin", to_pin), ("To tap", to_tap)]:
            if packages:
                info(f"{label}: {', '.join([str(p) for p in packages])}")
        return all_installed

    def install(self, packages: list[BrewPackageConfig]) -> None:
        to_install, to_pin, to_tap = self._get_changes_to_apply(packages)
        for tap_name in to_tap:
            info(f"Adding tap: {tap_name}")
            self._install_cmd(["brew", "tap", tap_name])

        if to_install:
            full_install_names = [p.fully_specified_name for p in packages]
            info(f"Installing {full_install_names}...")
            self._install_cmd(["brew", "install", *full_install_names])

        if to_pin:
            pin_names = [p.name for p in to_pin]
            info(f"Pinning {pin_names}...")
            self._install_cmd(["brew", "pin", *pin_names])
