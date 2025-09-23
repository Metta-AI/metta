import os
import platform
import shutil
import stat
import tempfile
from pathlib import Path

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, warning


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

    def _install_helm_if_missing(self, non_interactive: bool) -> None:
        if shutil.which("helm"):
            return

        info("Helm not found. Installing Helm...")

        if shutil.which("brew"):
            self.run_command(["brew", "install", "helm"], capture_output=False, non_interactive=non_interactive)
            return

        if platform.system() == "Linux":
            self._install_helm_via_script(non_interactive)
            return

        warning(
            "Helm is not installed and automatic installation is unsupported on this platform. "
            "Please install Helm manually and re-run metta install."
        )
        raise FileNotFoundError("Helm binary not found in PATH")

    def _install_helm_via_script(self, non_interactive: bool) -> None:
        install_script_url = "https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3"
        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = Path(tmp_dir) / "get_helm.sh"

            self.run_command(
                ["curl", "-fsSL", "-o", str(script_path), install_script_url],
                capture_output=False,
                non_interactive=non_interactive,
            )

            script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

            install_dir = Path.home() / ".local" / "bin"
            install_dir.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env.update({"USE_SUDO": "0", "HELM_INSTALL_DIR": str(install_dir)})
            env["PATH"] = f"{install_dir}:{env.get('PATH', '')}"

            self.run_command(
                ["bash", str(script_path)],
                capture_output=False,
                env=env,
                non_interactive=non_interactive,
            )

        info("Helm installation completed")

        if not shutil.which("helm"):
            warning(
                "Helm installer completed but the binary is still not on PATH. Add ~/.local/bin to your PATH or "
                "install Helm manually."
            )
            raise FileNotFoundError("Helm binary not found in PATH after installation")

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
        self._install_helm_if_missing(non_interactive)

        info("Setting up helm plugins...")

        installed_plugins = self.get_installed_plugins()
        for plugin, url in self.HELM_PLUGINS.items():
            if plugin not in installed_plugins:
                self.run_command(["helm", "plugin", "install", url])
