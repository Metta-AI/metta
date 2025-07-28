import subprocess
import tempfile
from pathlib import Path

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, info, success


@register_module
class FilterRepoSetup(SetupModule):
    install_once = True

    @property
    def name(self) -> str:
        return "filter-repo"

    @property
    def description(self) -> str:
        return "git-filter-repo tool for extracting repository subsets"

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("filter-repo")

    def check_installed(self) -> bool:
        """Check if git-filter-repo is installed."""
        try:
            result = subprocess.run(
                ["git", "filter-repo", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def install(self) -> None:
        """Install git-filter-repo."""
        info("Installing git-filter-repo...")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            script_path = temp_path / "git-filter-repo"
            
            # Download the script
            info("⬇️  Downloading git-filter-repo...")
            try:
                subprocess.run([
                    "curl", "-o", str(script_path),
                    "https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo"
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                error("Failed to download git-filter-repo")
                raise
            
            # Make it executable
            script_path.chmod(0o755)
            
            # Try to install to appropriate location
            install_locations = []
            user_bin = Path.home() / ".local" / "bin"
            
            # Check if user bin exists and is in PATH
            if user_bin.exists() and str(user_bin) in subprocess.run(
                ["bash", "-c", "echo $PATH"], capture_output=True, text=True
            ).stdout:
                install_locations.append(user_bin)
            
            # Add system locations
            install_locations.extend([Path("/usr/local/bin"), Path("/usr/bin")])
            
            installed = False
            for location in install_locations:
                if not location.exists():
                    continue
                    
                try:
                    dest_path = location / "git-filter-repo"
                    
                    # Check if we need sudo
                    if location in [Path("/usr/local/bin"), Path("/usr/bin")]:
                        # Try with sudo
                        info(f"📁 Installing to {location} (may require sudo)...")
                        subprocess.run([
                            "sudo", "cp", str(script_path), str(dest_path)
                        ], check=True)
                    else:
                        # Direct copy
                        info(f"📁 Installing to {location}...")
                        subprocess.run([
                            "cp", str(script_path), str(dest_path)
                        ], check=True)
                    
                    installed = True
                    success("✅ git-filter-repo installed successfully!")
                    break
                    
                except subprocess.CalledProcessError:
                    continue
            
            if not installed:
                error("Failed to install git-filter-repo automatically")
                error("")
                error("Please install manually:")
                error("  curl -O https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo")
                error("  chmod +x git-filter-repo")
                error("  sudo mv git-filter-repo /usr/local/bin/")
                raise RuntimeError("Installation failed")

    def run(self, args: list[str]) -> None:
        """Run filter-repo commands via metta."""
        if not args:
            error("Usage: metta run filter-repo <filter|inspect|push> ...")
            error("")
            error("This runs the sync_package.py script. Examples:")
            error("  metta run filter-repo filter . mettagrid/ mettascope/")
            error("  metta run filter-repo inspect /tmp/filtered-repo-xyz/filtered")
            error("  metta run filter-repo push /tmp/filtered-repo-xyz/filtered git@github.com:org/repo.git")
            return
        
        # Run the sync_package.py script with provided arguments
        script_path = self.repo_root / "devops" / "git" / "sync_package.py"
        if not script_path.exists():
            error(f"sync_package.py not found at {script_path}")
            return
        
        try:
            subprocess.run([str(script_path)] + args, check=True)
        except subprocess.CalledProcessError as e:
            error(f"Command failed with exit code {e.returncode}")