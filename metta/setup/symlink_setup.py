import os
import subprocess
from pathlib import Path
from typing import Optional

from metta.setup.utils import error, info, success, warning


class PathSetup:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.wrapper_script = repo_root / "metta" / "setup" / "installer" / "bin" / "metta"
        self.local_bin = Path.home() / ".local" / "bin"
        self.target_symlink = self.local_bin / "metta"

    def _local_bin_is_in_path(self) -> bool:
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        return str(self.local_bin) in path_dirs

    def _check_existing_metta(self) -> Optional[str]:
        if not self.target_symlink.exists():
            return None

        if self.target_symlink.is_symlink():
            target = self.target_symlink.resolve()
            if target == self.wrapper_script.resolve():
                return "ours"

        return "other"

    def setup_path(self, force: bool = False) -> None:
        self.wrapper_script.chmod(0o755)

        # Create ~/.local/bin if it doesn't exist
        try:
            self.local_bin.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            error(f"Failed to create {self.local_bin}: {e}")
            info("Please create this directory manually and re-run the installer.")
            return

        existing = self._check_existing_metta()

        if existing == "ours":
            success("metta is already installed and linked correctly.")
            return
        elif existing == "other":
            if force:
                info(f"Replacing existing metta command at {self.target_symlink}")
                try:
                    self.target_symlink.unlink()
                except Exception as e:
                    error(f"Failed to remove existing file: {e}")
                    return
            else:
                warning(f"A 'metta' command already exists at {self.target_symlink}")
                info("Not overwriting existing command.")
                info("Run with --force if you want to replace it.")
                return

        try:
            self.target_symlink.symlink_to(self.wrapper_script)
            success(f"Created symlink: {self.target_symlink} → {self.wrapper_script}")
        except Exception as e:
            error(f"Failed to create symlink: {e}")
            info("You can create it manually with:")
            info(f"  ln -s {self.wrapper_script} {self.target_symlink}")
            return

        if self._local_bin_is_in_path():
            success("metta is now available globally!")
            info("You can run: metta --help")
        else:
            info("")
            warning(f"{self.local_bin} is not in your PATH")
            info("To use metta globally, add this to your shell profile:")
            info('  export PATH="$HOME/.local/bin:$PATH"')
            info("")
            info("For example:")
            info("  echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc")
            info("  source ~/.bashrc")

    def check_installation(self) -> bool:
        existing = self._check_existing_metta()
        if existing != "ours":
            return False
        try:
            result = subprocess.run(["which", "metta"], capture_output=True, text=True)
            if result.returncode == 0:
                # Check if it points to our wrapper script
                found_path = Path(result.stdout.strip())
                return found_path.resolve() == self.wrapper_script.resolve()
        except Exception:
            pass

        return False
