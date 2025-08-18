from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module


@register_module
class MettaScopeSetup(SetupModule):
    @property
    def description(self) -> str:
        return "MettaScope visualization and replay tools"

    def dependencies(self) -> list[str]:
        return ["nodejs"]

    @property
    def setup_script_location(self) -> str:
        return "mettascope/install.sh"

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("mettascope")

    def check_installed(self) -> bool:
        mettascope_dir = self.repo_root / "mettascope"
        node_modules = mettascope_dir / "node_modules"
        return node_modules.exists()
