import os
import platform
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

        if not self._script_exists("turbo"):
            return False

        return True

    def _check_pnpm(self) -> bool:
        """Check if pnpm is working."""
        try:
            env = os.environ.copy()
            env["NODE_NO_WARNINGS"] = "1"
            result = subprocess.run(
                ["pnpm", "--version"],
                capture_output=True,
                text=True,
                env=env,
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
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
        info("Setting up pnpm...")

        if not self._check_pnpm():
            # Try to enable corepack with automatic cleanup
            if not self._enable_corepack_with_cleanup():
                raise RuntimeError("Failed to set up pnpm via corepack")

        def _set_pnpm_home_now(value: str) -> None:
            os.environ["PNPM_HOME"] = value
            os.environ["PATH"] = f"{value}:{os.environ['PATH']}"  # pnpm complains if PNPM_HOME is not in PATH
            info("PNPM_HOME configured. Restart your shell to apply.")

        if not os.environ.get("PNPM_HOME"):
            # We need to setup pnpm before we can install turbo globally
            # This command will update the user's `~/.bashrc` or `~/.zshrc`.
            self.run_command(["pnpm", "setup"], capture_output=False)

            # PNPM_HOME configuration is in the user's shell profile, but we need to set it now.
            # Apply some heuristics to detect the correct directory.
            #
            # Note: we could run a new temporary shell script, print env from it and capture, but that might be more
            # error prone.
            if platform.system() == "Darwin":
                _set_pnpm_home_now(os.path.expanduser("~/Library/pnpm"))
            elif os.path.exists(os.path.expanduser("~/.pnpm")):
                _set_pnpm_home_now(os.path.expanduser("~/.pnpm"))

        if os.environ.get("PNPM_HOME"):
            info("Installing turbo...")
            self.run_command(["pnpm", "install", "--global", "turbo"], capture_output=False)
        else:
            warning("Failed to detect PNPM_HOME dir, skipping global turbo install")

        info("Installing dependencies...")
        # pnpm install with frozen lockfile to avoid prompts
        self.run_command(["pnpm", "install", "--frozen-lockfile"], capture_output=False)
