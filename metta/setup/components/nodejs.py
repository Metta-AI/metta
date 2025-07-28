import os
import re
import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, warning


@register_module
class NodejsSetup(SetupModule):
    @property
    def description(self) -> str:
        return "Node.js infrastructure - pnpm and turborepo"

    def is_applicable(self) -> bool:
        return self.config.is_component_enabled("nodejs")

    def _script_exists(self, script: str) -> bool:
        try:
            self.run_command(["which", script], capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def check_installed(self) -> bool:
        if not (self.repo_root / "node_modules").exists():
            return False

        if not self._check_pnpm():
            return False

        # Turbo can be available via npx even if not globally installed
        # So we don't require it to be in PATH
        return True

    def _check_pnpm(self) -> bool:
        """Check if pnpm is working."""
        try:
            env = os.environ.copy()
            env["NODE_NO_WARNINGS"] = "1"
            env["COREPACK_ENABLE_STRICT"] = "0"  # Avoid download prompts
            self.run_command(["pnpm", "--version"], capture_output=True, env=env)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _enable_corepack_with_cleanup(self):
        """Enable corepack, removing dead symlinks as needed."""
        info("Attempting to enable corepack...")
        for _attempt in range(10):
            try:
                # Try to enable corepack with timeout
                # Set COREPACK_ENABLE_STRICT=0 to avoid prompts
                env = os.environ.copy()
                env["COREPACK_ENABLE_STRICT"] = "0"
                subprocess.run(
                    ["corepack", "enable"],
                    capture_output=True,
                    text=True,
                    timeout=30,  # 30 second timeout
                    check=True,
                    env=env,
                )
                info("Corepack enabled successfully")
                return True
            except subprocess.CalledProcessError as e:
                # Get error output from stdout or stderr
                error_output = ""
                if hasattr(e, "stdout") and e.stdout:
                    error_output += e.stdout
                if hasattr(e, "stderr") and e.stderr:
                    error_output += e.stderr

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
            except subprocess.TimeoutExpired as e:
                warning("Corepack enable timed out after 30 seconds")
                raise RuntimeError("Corepack enable timed out") from e

        warning("Failed to enable corepack after removing dead symlinks")
        return False

    def install(self) -> None:
        info("Setting up pnpm...")

        if not self._check_pnpm():
            # Try to enable corepack with automatic cleanup
            if not self._enable_corepack_with_cleanup():
                raise RuntimeError("Failed to set up pnpm via corepack")

        # Run pnpm setup to ensure global bin directory exists
        info("Running pnpm setup...")
        try:
            env = os.environ.copy()
            env["NODE_NO_WARNINGS"] = "1"
            env["COREPACK_ENABLE_STRICT"] = "0"
            self.run_command(["pnpm", "setup"], capture_output=False, env=env)
            info("pnpm setup completed. You may need to restart your shell for PATH changes to take effect.")
        except subprocess.CalledProcessError:
            warning("pnpm setup failed - continuing anyway")

        info("Installing turbo...")
        # Use npx to run turbo instead of global install to avoid PATH issues
        # First try global install, but if it fails, we'll use npx
        try:
            env = os.environ.copy()
            env["NODE_NO_WARNINGS"] = "1"
            env["COREPACK_ENABLE_STRICT"] = "0"
            self.run_command(["pnpm", "install", "--global", "turbo"], capture_output=False, env=env)
        except subprocess.CalledProcessError:
            info("Global install failed, turbo will be available via npx")

        info("Installing dependencies...")
        # pnpm install with frozen lockfile to avoid prompts
        env = os.environ.copy()
        env["NODE_NO_WARNINGS"] = "1"
        env["COREPACK_ENABLE_STRICT"] = "0"
        self.run_command(["pnpm", "install", "--frozen-lockfile"], capture_output=False, env=env)
