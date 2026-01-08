import shutil

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info


@register_module
class HelmSetup(SetupModule):
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
            if plugin not in installed_plugins:
                # The `--verify=false` flag is a temporary workaround for a Helm bug/incompatibility in plugin signature
                #  verification introduced in v4. See https://github.com/helm/helm/issues/31490#issuecomment-3548420699
                #  Worth revisiting when this issue is fixed
                self.run_command(["helm", "plugin", "install", "--verify=false", url])
