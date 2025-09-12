import subprocess
from pathlib import Path

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, info, success


@register_module
class TribalSetup(SetupModule):
    @property
    def description(self) -> str:
        return "Tribal environment and bindings"

    def check_installed(self) -> bool:
        """Check if tribal package is installed and bindings exist."""
        try:
            # Check if tribal package is installed by trying to import directly
            # (avoid uv run subprocess which has dependency resolution issues with local packages)
            try:
                import metta.tribal
                package_installed = True
            except ImportError:
                package_installed = False
                
            if not package_installed:
                return False
                
            # Check if bindings exist
            project_root = Path(__file__).parent.parent.parent.parent
            tribal_dir = project_root / "tribal" 
            bindings_dir = tribal_dir / "bindings" / "generated"
            
            if not bindings_dir.exists():
                return False
                
            library_files = list(bindings_dir.glob("libtribal.*"))
            python_binding = bindings_dir / "tribal.py"
            
            return len(library_files) > 0 and python_binding.exists()
            
        except Exception:
            return False

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        """Install tribal package and build bindings."""
        project_root = Path(__file__).parent.parent.parent.parent
        tribal_dir = project_root / "tribal"
        
        if not tribal_dir.exists():
            error("Tribal directory not found")
            return
            
        # Install tribal package in development mode
        info("Installing tribal package...")
        self.run_command(
            ["uv", "pip", "install", "-e", str(tribal_dir)],
            non_interactive=non_interactive
        )
        
        # Build tribal bindings
        info("Building tribal bindings...")
        self.run_command(
            ["nimble", "bindings"],
            cwd=str(tribal_dir),
            non_interactive=non_interactive
        )
        
        success("Tribal environment and bindings installed")