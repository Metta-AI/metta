import shutil
import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, warning

VERSIONED_NODE_PATHS = ("/opt/homebrew/opt/node@", "/home/linuxbrew/.linuxbrew/opt/node@")


@register_module
class JsToolchainSetup(SetupModule):
    @property
    def name(self) -> str:
        return "js-toolchain"

    @property
    def description(self) -> str:
        return "JavaScript toolchain (pnpm, turborepo)"

    def dependencies(self) -> list[str]:
        return ["system"]

    def _cmd_exists(self, script: str) -> bool:
        return shutil.which(script) is not None

    def _warn_if_versioned_node(self) -> None:
        """Warn if corepack/pnpm are from a pinned node version (e.g. node@24)."""
        for cmd in ("corepack", "pnpm"):
            path = shutil.which(cmd)
            if path and any(path.startswith(p) for p in VERSIONED_NODE_PATHS):
                warning(
                    f"{cmd} is from a node version we are migrating away from: {path}\n"
                    "  To clean up: brew uninstall node@24, then re-run metta install"
                )

    def check_installed(self) -> bool:
        if not all(self._cmd_exists(cmd) for cmd in ("node", "corepack", "pnpm", "turbo")):
            return False

        if not (self.repo_root / "node_modules").exists():
            return False

        return True

    def _enable_corepack(self) -> None:
        try:
            self.run_command(["corepack", "enable"], capture_output=True, check=True)
            info("Corepack enabled successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"corepack enable failed: {e.output}") from e

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        info("Setting up JavaScript toolchain...")
        self._warn_if_versioned_node()

        if force:
            info("Uninstalling pnpm first...")
            self.run_command(["corepack", "disable"])

        # Enable corepack to make pnpm available
        if not self._cmd_exists("pnpm"):
            if not self._cmd_exists("corepack"):
                raise RuntimeError("corepack not found. Run 'metta install system' first.")
            self._enable_corepack()

        # pnpm setup updates shell profile so pnpm globals are on PATH in new shells
        info("Running pnpm setup...")
        self.run_command(["pnpm", "setup"], capture_output=False)

        info("Installing turbo globally...")
        self.run_command(["pnpm", "install", "--global", "turbo"], capture_output=False)

        info("Installing project dependencies...")
        # pnpm install with frozen lockfile to avoid prompts
        self.run_command(["pnpm", "install", "--frozen-lockfile"], capture_output=False, cwd=self.repo_root)

        info("JS toolchain setup completed. Restart your shell if 'pnpm' or 'turbo' aren't found.")
