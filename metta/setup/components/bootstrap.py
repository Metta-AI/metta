"""Bootstrap setup module.

Installs critical bootstrap dependencies needed before other components:
- bazel (via bazelisk)
- nimby + nim
- git, g++ (via system package manager)
"""

from typing_extensions import override

from metta.setup.components.base import SetupModule
from metta.setup.components.system_packages import bootstrap as bootstrap_module
from metta.setup.registry import register_module
from metta.setup.utils import success


@register_module
class BootstrapSetup(SetupModule):
    always_required = True

    @property
    @override
    def name(self) -> str:
        return "bootstrap"

    @property
    @override
    def description(self) -> str:
        return "Bootstrap dependencies (bazel, nimby, nim, git, g++)"

    @override
    def dependencies(self) -> list[str]:
        return []

    @override
    def check_installed(self) -> bool:
        return bootstrap_module.check_bootstrap_deps()

    @override
    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        bootstrap_module.install_bootstrap_deps(self.run_command, non_interactive=non_interactive)
        success("Bootstrap dependencies installed")
