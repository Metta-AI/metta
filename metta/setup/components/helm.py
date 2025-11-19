import shutil

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, info, warning


@register_module
class HelmSetup(SetupModule):
    install_once = True

    HELM_PLUGINS = {
        "diff": "https://github.com/databus23/helm-diff",
        "helm-git": "https://github.com/aslafy-z/helm-git",
    }

    @property
    def description(self) -> str:
        return "Helm plugins"

    def dependencies(self) -> list[str]:
        return ["system"]  # Ensure helm is installed

    def get_installed_plugins(self) -> list[str]:
        plugins_result = self.run_command(["helm", "plugin", "list"])
        plugins = [
            line.split("\t")[0].strip() for line in plugins_result.stdout.split("\n") if not line.startswith("NAME")
        ]
        return plugins

    def check_installed(self) -> bool:
        if not shutil.which("helm"):
            return False

        installed_plugins = self.get_installed_plugins()
        for plugin in self.HELM_PLUGINS.keys():
            if plugin not in installed_plugins:
                return False
        return True

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        info("Setting up helm plugins...")

        installed_plugins = self.get_installed_plugins()
        for plugin, url in self.HELM_PLUGINS.items():
            # Uninstall existing plugin if force is enabled
            if force and plugin in installed_plugins:
                info(f"Uninstalling existing helm plugin: {plugin}")
                uninstall_result = self.run_command(
                    ["helm", "plugin", "uninstall", plugin], capture_output=True, check=False
                )
                if uninstall_result.returncode != 0:
                    warning(f"Failed to uninstall helm plugin '{plugin}': {uninstall_result.stderr}")

            if plugin not in installed_plugins:
                info(f"Installing helm plugin: {plugin}")
                install_cmd = ["helm", "plugin", "install", url]
                if force:
                    install_cmd.extend(["--verify=false"])
                result = self.run_command(install_cmd, capture_output=True, check=False)

                if result.returncode == 0:
                    info(f"Installed helm plugin '{plugin}'.")
                    continue

                # Show error details
                if result.stderr:
                    error(f"Error installing helm plugin '{plugin}': {result.stderr}")
                else:
                    error(f"Failed to install helm plugin '{plugin}' (exit code: {result.returncode})")
                warning(f"Skipping helm plugin '{plugin}' installation")
