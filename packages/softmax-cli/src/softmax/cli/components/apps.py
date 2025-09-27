import os
import platform
from pathlib import Path

import yaml
from pydantic.main import BaseModel
from typing_extensions import override

from softmax.cli.components.base import SetupModule
from softmax.cli.components.system import get_package_installer
from softmax.cli.components.system_packages.installers.brew import BrewInstaller
from softmax.cli.registry import register_module


class AppConfig(BaseModel):
    cask: str | None = None
    alternate_app_path: str | None = None


class SystemAppsConfig(BaseModel):
    apps: dict[str, AppConfig]


def get_system_apps_config() -> SystemAppsConfig:
    config_path = Path(__file__).resolve().parent / "system_packages" / "apps.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return SystemAppsConfig(**data)


@register_module
class AppsSetup(SetupModule):
    @property
    def description(self) -> str:
        return "Applications"

    def dependencies(self) -> list[str]:
        return ["system"]

    @override
    def check_installed(self) -> bool:
        return not bool(self._get_uninstalled_apps())

    def _get_uninstalled_apps(self) -> list[AppConfig]:
        apps = list(get_system_apps_config().apps.values())
        installer = self._get_brew_installer_if_available()
        installed_casks = installer.get_installed_casks() if installer else []

        uninstalled: list[AppConfig] = []
        for app in apps:
            if app.cask and app.cask in installed_casks:
                continue
            elif app.alternate_app_path and os.path.exists(app.alternate_app_path):
                continue
            else:
                uninstalled.append(app)
        return uninstalled

    def _get_brew_installer_if_available(self) -> BrewInstaller | None:
        installer = get_package_installer()
        if not installer or not isinstance(installer, BrewInstaller):
            return None
        return installer

    def is_applicable(self) -> bool:
        installer = self._get_brew_installer_if_available()
        return platform.system() == "Darwin" and installer is not None

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        installer = self._get_brew_installer_if_available()
        if not installer:
            return
        uninstalled = self._get_uninstalled_apps()
        cask_installable = [app_cfg.cask for app_cfg in uninstalled if app_cfg.cask]
        if cask_installable:
            installer.install_casks(cask_installable)
