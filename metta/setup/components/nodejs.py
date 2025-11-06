import os
import platform
import re
import shutil
import subprocess

import metta.setup.components.base
import metta.setup.registry
import metta.setup.utils


@metta.setup.registry.register_module
class NodejsSetup(metta.setup.components.base.SetupModule):
    @property
    def description(self) -> str:
        return "Node.js infrastructure - pnpm and turborepo"

    def dependencies(self) -> list[str]:
        return ["system"]  # Ensure Node.js/corepack is installed before running pnpm setup

    def _cmd_exists(self, script: str) -> bool:
        return shutil.which(script) is not None

    def check_installed(self) -> bool:
        if not (self.repo_root / "node_modules").exists():
            return False

        if not self._cmd_exists("pnpm"):
            return False

        if not self._cmd_exists("turbo"):
            return False

        return True

    def _check_pnpm(self) -> bool:
        """Check if pnpm is available."""
        return shutil.which("pnpm") is not None

    def _enable_corepack_with_cleanup(self):
        """Enable corepack, removing dead symlinks as needed."""
        for _ in range(10):
            try:
                # Try to enable corepack
                self.run_command(["corepack", "enable"], capture_output=True, check=True)
                metta.setup.utils.info("Corepack enabled successfully")
                return True
            except subprocess.CalledProcessError as e:
                error_output = e.output
                # Look for EEXIST error with file path
                match = re.search(r"EEXIST: file already exists, symlink .* -> '([^']+)'", error_output)
                if match:
                    conflicting_path = match.group(1)
                    metta.setup.utils.warning(f"Removing dead symlink: {conflicting_path}")
                    try:
                        if os.path.islink(conflicting_path):
                            os.remove(conflicting_path)
                            metta.setup.utils.info(f"Removed conflicting file: {conflicting_path}")
                        else:
                            metta.setup.utils.warning(f"Conflicting file not found: {conflicting_path}")
                    except OSError as rm_err:
                        metta.setup.utils.warning(f"Failed to remove {conflicting_path}: {rm_err}")
                        raise RuntimeError(f"Cannot remove conflicting file: {conflicting_path}") from rm_err
                    continue
                else:
                    # Not an EEXIST error or couldn't parse it
                    metta.setup.utils.warning(f"Corepack enable failed: {error_output}")
                    raise

        metta.setup.utils.warning("Failed to enable corepack after removing dead symlinks")
        return False

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        metta.setup.utils.info("Setting up pnpm...")

        if force:
            metta.setup.utils.info("Uninstalling pnpm first...")
            self.run_command(["corepack", "disable"])

        # First enable corepack to make pnpm command available
        if not self._cmd_exists("pnpm"):
            if not self._enable_corepack_with_cleanup():
                raise RuntimeError("Failed to set up pnpm via corepack")

        # Let pnpm setup itself - this handles shell profile configuration
        metta.setup.utils.info("Running pnpm setup to configure shell profiles...")
        try:
            self.run_command(["pnpm", "setup"], capture_output=False)
            metta.setup.utils.info("pnpm setup completed successfully")
        except subprocess.CalledProcessError as e:
            # pnpm setup can fail if there's already a config, but that's usually OK
            metta.setup.utils.warning(f"pnpm setup returned non-zero exit code: {e}. Continuing...")

        # Set PNPM_HOME for current process if pnpm setup configured it
        # Use platform-specific default locations that match pnpm setup behavior
        if platform.system() == "Darwin":
            pnpm_home = os.path.expanduser("~/Library/pnpm")
        else:
            pnpm_home = os.path.expanduser("~/.local/share/pnpm")

        if os.path.exists(pnpm_home):
            metta.setup.utils.info(f"Setting PNPM_HOME to {pnpm_home} for current process")
            os.environ["PNPM_HOME"] = pnpm_home
            # Add to PATH for current process
            current_path = os.environ.get("PATH", "")
            if pnpm_home not in current_path:
                os.environ["PATH"] = f"{pnpm_home}:{current_path}"

        # Install global turbo if pnpm is working
        if self._cmd_exists("pnpm"):
            metta.setup.utils.info("Installing turbo globally...")
            self.run_command(["pnpm", "install", "--global", "turbo"], capture_output=False)
        else:
            metta.setup.utils.warning("pnpm not working properly, skipping global turbo install")

        metta.setup.utils.info("Installing project dependencies...")
        # pnpm install with frozen lockfile to avoid prompts
        self.run_command(["pnpm", "install", "--frozen-lockfile"], capture_output=False, cwd=self.repo_root)

        metta.setup.utils.info("Nodejs setup completed! You may need to restart your shell to use global packages.")
