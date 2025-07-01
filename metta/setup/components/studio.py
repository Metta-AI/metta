from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, success, warning


@register_module
class StudioSetup(SetupModule):
    @property
    def name(self) -> str:
        return "studio"

    @property
    def description(self) -> str:
        return "Studio frontend development"

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("studio")

    def check_installed(self) -> bool:
        studio_dir = self.repo_root / "studio"
        return (studio_dir / "node_modules").exists()

    def install(self) -> None:
        info("Setting up Studio frontend...")

        studio_dir = self.repo_root / "studio"
        if not studio_dir.exists():
            warning("Studio directory not found")
            return

        self.run_command(["corepack", "enable"])
        self.run_command(["pnpm", "install"], cwd=studio_dir)

        success("Studio frontend installed")
