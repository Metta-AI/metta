from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, success


@register_module
class DevtoolsSetup(SetupModule):
    @property
    def name(self) -> str:
        return "devtools"

    @property
    def description(self) -> str:
        return "Developer tools"

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("devtools")

    def check_installed(self) -> bool:
        # TODO
        return True

    def install(self) -> None:
        info("Setting up developer tools...")
        # TODO
        success("Devtools installed")
