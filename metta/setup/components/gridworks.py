from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, success, warning


@register_module
class GridworksSetup(SetupModule):
    @property
    def name(self) -> str:
        return "gridworks"

    @property
    def description(self) -> str:
        return "Gridworks frontend development"

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("gridworks")

    def check_installed(self) -> bool:
        gridworks_dir = self.repo_root / "gridworks"
        return (gridworks_dir / "node_modules").exists()

    def install(self) -> None:
        info("Setting up Gridworks frontend...")

        gridworks_dir = self.repo_root / "gridworks"
        if not gridworks_dir.exists():
            warning("Gridworks directory not found")
            return

        # Corepack enable with auto-yes (no prompts needed, runs with current user)
        self.run_command(["corepack", "enable"], capture_output=False)

        # pnpm install with frozen lockfile to avoid prompts
        self.run_command(["pnpm", "install", "--frozen-lockfile"], cwd=gridworks_dir, capture_output=False)

        success("Gridworks frontend installed")
