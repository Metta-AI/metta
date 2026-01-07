import os
import shutil
from pathlib import Path

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, warning

NVM_VERSION = "v0.40.1"
NVM_INSTALL_URL = f"https://raw.githubusercontent.com/nvm-sh/nvm/{NVM_VERSION}/install.sh"
BREW_NODE_PATHS = [
    Path("/opt/homebrew/opt/node@24"),
    Path("/opt/homebrew/opt/node@22"),
    Path("/opt/homebrew/opt/node"),
    Path("/home/linuxbrew/.linuxbrew/opt/node@24"),
    Path("/home/linuxbrew/.linuxbrew/opt/node@22"),
    Path("/home/linuxbrew/.linuxbrew/opt/node"),
]


@register_module
class JsToolchainSetup(SetupModule):
    @property
    def name(self) -> str:
        return "js-toolchain"

    @property
    def description(self) -> str:
        return "JavaScript toolchain (nvm, node, pnpm, turborepo)"

    def dependencies(self) -> list[str]:
        return ["system"]

    @property
    def _nvm_dir(self) -> Path:
        return Path(os.environ.get("NVM_DIR", Path.home() / ".nvm"))

    def _nvm_installed(self) -> bool:
        return (self._nvm_dir / "nvm.sh").exists()

    def _node_available(self) -> bool:
        return shutil.which("node") is not None

    def _get_brew_node_installations(self) -> list[Path]:
        return [p for p in BREW_NODE_PATHS if p.exists()]

    def check_installed(self) -> bool:
        if not self._nvm_installed():
            return False

        if not all(shutil.which(cmd) for cmd in ("node", "corepack", "pnpm", "turbo")):
            return False

        if not (self.repo_root / "node_modules").exists():
            return False

        return True

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        info("Setting up JavaScript toolchain...")

        brew_nodes = self._get_brew_node_installations()
        if brew_nodes:
            warning("Found brew-installed Node.js that may conflict with nvm:")
            for p in brew_nodes:
                warning(f"  {p}")
            warning("Recommend: brew uninstall node node@22 node@24")

        if not self._nvm_installed():
            info(f"Installing nvm {NVM_VERSION}...")
            self.run_command(
                ["bash", "-c", f"curl -o- {NVM_INSTALL_URL} | bash"],
                capture_output=False,
                env={"NVM_DIR": str(self._nvm_dir)},
            )
            info("Installing Node.js 24 via nvm...")
            self.run_command(
                ["bash", "-c", f"source {self._nvm_dir}/nvm.sh && nvm install 24"],
                capture_output=False,
            )
            info("nvm and Node.js installed. Restart your shell, then run: metta install")
            raise SystemExit(1)

        if not self._node_available():
            info("Node.js not in PATH. Restart your shell, then run: metta install")
            raise SystemExit(1)

        if not shutil.which("corepack"):
            info("Installing corepack...")
            self.run_command(["npm", "install", "-g", "corepack"], capture_output=False)

        if not shutil.which("pnpm"):
            info("Enabling corepack...")
            self.run_command(["corepack", "enable"], capture_output=False)

        info("Running pnpm setup...")
        result = self.run_command(["pnpm", "setup"], capture_output=True, check=False)
        if result.returncode != 0:
            warning("pnpm setup returned non-zero (profile may already be configured)")

        if not shutil.which("turbo"):
            info("Installing turbo globally...")
            self.run_command(["pnpm", "install", "--global", "turbo"], capture_output=False)

        info("Installing project dependencies...")
        self.run_command(["pnpm", "install", "--frozen-lockfile"], capture_output=False, cwd=self.repo_root)

        info("JS toolchain setup completed.")
        if not shutil.which("turbo"):
            info("Note: Restart your shell for 'turbo' to be available in PATH.")
