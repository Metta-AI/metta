import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import warning


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

    def install(self) -> None:
        """Install MettaScope with timeout to prevent hanging."""
        script_path = self.repo_root / self.setup_script_location
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        try:
            # Run with timeout to prevent hanging on gen_atlas.py
            subprocess.run(
                ["bash", str(script_path)],
                cwd=self.repo_root,
                capture_output=False,
                text=True,
                timeout=300,  # 5 minute timeout
                check=True,
            )
        except subprocess.TimeoutExpired as e:
            warning("MettaScope installation timed out after 5 minutes")
            raise RuntimeError("MettaScope installation timed out - atlas generation may have failed") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"MettaScope installation failed with exit code {e.returncode}") from e
