import os
import re
import subprocess

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

    def _check_pnpm(self) -> bool:
        """Check if pnpm is working."""
        try:
            env = os.environ.copy()
            env["NODE_NO_WARNINGS"] = "1"
            self.run_command(["pnpm", "--version"], capture_output=True, env=env)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _enable_corepack_with_cleanup(self):
        """Enable corepack, removing dead symlinks as needed."""
        for _ in range(10):
            try:
                # Try to enable corepack
                self.run_command(["corepack", "enable"], capture_output=True, check=True)
                info("Corepack enabled successfully")
                return True
            except subprocess.CalledProcessError as e:
                error_output = e.output
                # Look for EEXIST error with file path
                match = re.search(r"EEXIST: file already exists, symlink .* -> '([^']+)'", error_output)
                if match:
                    conflicting_path = match.group(1)
                    warning(f"Removing dead symlink: {conflicting_path}")
                    try:
                        if os.path.islink(conflicting_path):
                            os.remove(conflicting_path)
                            info(f"Removed conflicting file: {conflicting_path}")
                        else:
                            warning(f"Conflicting file not found: {conflicting_path}")
                    except OSError as rm_err:
                        warning(f"Failed to remove {conflicting_path}: {rm_err}")
                        raise RuntimeError(f"Cannot remove conflicting file: {conflicting_path}") from rm_err
                    continue
                else:
                    # Not an EEXIST error or couldn't parse it
                    warning(f"Corepack enable failed: {error_output}")
                    raise

        warning("Failed to enable corepack after removing dead symlinks")
        return False

    def install(self) -> None:
        info("Setting up Gridworks frontend...")

        gridworks_dir = self.repo_root / "gridworks"
        if not gridworks_dir.exists():
            warning("Gridworks directory not found")
            return

        if not self._check_pnpm():
            # Try to enable corepack with automatic cleanup
            if not self._enable_corepack_with_cleanup():
                raise RuntimeError("Failed to set up pnpm via corepack")

        # Run pnpm install
        env = os.environ.copy()
        env["NODE_OPTIONS"] = "--no-deprecation"

        self.run_command(
            ["pnpm", "install", "--frozen-lockfile", "--silent"], cwd=gridworks_dir, capture_output=False, env=env
        )

        success("Gridworks frontend installed")
