import subprocess

from metta.setup.components.system_packages.installers.base import PackageInstaller
from metta.setup.components.system_packages.types import AptPackageConfig
from metta.setup.utils import info


class AptInstaller(PackageInstaller[AptPackageConfig]):
    def __init__(self):
        self._apt_updated = False

    @property
    def name(self) -> str:
        return "apt"

    def is_available(self) -> bool:
        result = subprocess.run(["which", "apt"], check=False)
        return result.returncode == 0

    def _get_list_cmd(self, cmd: list[str]) -> list[str]:
        result = subprocess.run(cmd, text=True, check=True, capture_output=True)
        lines = result.stdout.strip().split("\n")
        return [line.split("\t")[0] for line in lines if "\tinstall" in line]

    def _install_cmd(self, cmd: list[str]) -> None:
        subprocess.run(cmd, text=True, check=True, capture_output=False)

    def _get_installed_packages(self) -> list[str]:
        return self._get_list_cmd(["dpkg", "--get-selections"])

    def check_installed(self, packages: list[AptPackageConfig]) -> bool:
        """Returns True when no changes are required."""
        installed = self._get_installed_packages()
        package_names = [p.name for p in packages]
        to_install = [pkg for pkg in package_names if pkg not in installed]
        return len(to_install) == 0

    def install(self, packages: list[AptPackageConfig]) -> None:
        # Update package list first (once per session)
        if not self._apt_updated:
            info("Updating apt package list...")
            self._install_cmd(["sudo", "apt-get", "update"])
            self._apt_updated = True

        installed = self._get_installed_packages()
        package_names = [p.name for p in packages]
        to_install = [pkg for pkg in package_names if pkg not in installed]

        if to_install:
            info(f"Installing {', '.join(to_install)}...")
            self._install_cmd(["sudo", "apt-get", "install", "-y", *to_install])
