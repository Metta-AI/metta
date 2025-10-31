from __future__ import annotations

import shutil

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import info, warning


@register_module
class VscodeextensionsSetup(SetupModule):
    """Ensure required VS Code extensions are installed."""

    install_once = True
    _required_extensions: list[str] = ["emeraldwalk.runonsave"]

    def __init__(self) -> None:
        super().__init__()
        self._missing: list[str] = []

    @property
    def description(self) -> str:
        return "VS Code extensions required for developer ergonomics"

    def dependencies(self) -> list[str]:
        # Requires basic tooling to be installed first
        return ["system"]

    def _find_code_command(self) -> str | None:
        for candidate in ("code", "code-insiders"):
            if (cmd := shutil.which(candidate)) is not None:
                return cmd
        return None

    def _list_installed_extensions(self, code_cmd: str) -> set[str]:
        result = self.run_command([code_cmd, "--list-extensions"], capture_output=True, check=False)
        if result.returncode != 0:
            warning("Unable to list VS Code extensions; assuming none are installed.")
            return set()

        return {line.strip() for line in result.stdout.splitlines() if line.strip()}

    def check_installed(self) -> bool:
        code_cmd = self._find_code_command()
        if not code_cmd:
            warning("VS Code command-line interface (code) not found; skipping extension install.")
            return True

        installed = self._list_installed_extensions(code_cmd)
        self._missing = [ext for ext in self._required_extensions if ext not in installed]
        return not self._missing

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        self._non_interactive = non_interactive

        code_cmd = self._find_code_command()
        if not code_cmd:
            warning(
                "VS Code CLI (code) not found on PATH. "
                "Install Visual Studio Code and enable the 'Shell Command: Install code command' option."
            )
            return

        if not self._missing or force:
            # Recalculate missing extensions in case install() is called without check_installed first
            installed = self._list_installed_extensions(code_cmd)
            self._missing = [ext for ext in self._required_extensions if ext not in installed]

        if not self._missing:
            info("Required VS Code extensions already installed.")
            return

        for extension in self._missing:
            info(f"Installing VS Code extension: {extension}")
            result = self.run_command(
                [code_cmd, "--install-extension", extension],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                warning(
                    f"Failed to install VS Code extension '{extension}'. "
                    f"Install it manually with: {code_cmd} --install-extension {extension}"
                )
            else:
                info(f"Installed VS Code extension '{extension}'.")

        # Reset missing list after attempt; subsequent checks will refresh it
        self._missing = []
