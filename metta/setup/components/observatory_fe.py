from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, success, warning


@register_module
class ObservatoryFeSetup(SetupModule):
    @property
    def name(self) -> str:
        return "observatory-fe"

    @property
    def description(self) -> str:
        return "Observatory frontend development"

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("observatory-fe")

    def check_installed(self) -> bool:
        observatory_dir = self.repo_root / "observatory"
        return (observatory_dir / "node_modules").exists()

    def install(self) -> None:
        info("Setting up Observatory frontend...")

        observatory_dir = self.repo_root / "observatory"
        if not observatory_dir.exists():
            warning("Observatory directory not found")
            return

        self.run_command(["pnpm", "install"], cwd=observatory_dir)

        success("Observatory frontend installed")
