from __future__ import annotations

import shutil

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, info, warning


@register_module
class IdeExtensions(SetupModule):
    """Ensure required VS Code / Cursor extensions are installed."""

    install_once = True
    _required_extensions: list[str] = ["emeraldwalk.runonsave"]

    @property
    def name(self) -> str:
        return "ide-extensions"

    @property
    def description(self) -> str:
        return "VS Code extensions required for developer ergonomics"

    def _candidate_commands(self) -> list[str]:
        return ["cursor", "code"]

    def _find_code_command(self) -> str | None:
        for candidate in self._candidate_commands():
            if (cmd := shutil.which(candidate)) is not None:
                return cmd
        return None

    def _list_installed_extensions(self, code_cmd: str) -> set[str]:
        result = self.run_command([code_cmd, "--list-extensions"], capture_output=True, check=False)
        if result.returncode != 0:
            warning("Unable to list VS Code extensions; assuming none are installed.")
            return set()

        return {line.strip() for line in result.stdout.splitlines() if line.strip()}

    def _is_applicable(self) -> bool:
        if self._find_code_command():
            return True

        warning("VS Code (code/cursor) CLI not detected; skipping VS Code extension installation. ")
        return False

    def check_installed(self) -> bool:
        code_cmd = self._find_code_command()
        if not code_cmd:
            return True

        installed = self._list_installed_extensions(code_cmd)
        missing = set(self._required_extensions) - installed
        return not missing

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        self._non_interactive = non_interactive

        code_cmd = self._find_code_command()
        if not code_cmd:
            return
        for extension in self._required_extensions:
            info(f"Installing VS Code extension: {extension}")
            install_cmd = [code_cmd, "--install-extension", extension]
            if force:
                install_cmd.append("--force")
            result = self.run_command(install_cmd, capture_output=True, check=False)

            if result.returncode == 0:
                info(f"Installed VS Code extension '{extension}'.")
                continue
            error(result.stderr)
            warning(f"Failed to install VS Code extension '{extension}'")
