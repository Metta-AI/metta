import os
import shutil

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, info, warning


@register_module
class IdeExtensions(SetupModule):
    """Ensure required VS Code / Cursor extensions are installed."""

    install_once = True
    _required_extensions: dict[str, str] = {
        "emeraldwalk.runonsave": "./metta/setup/components/extensions/RunOnSave-0.2.7.vsix"
    }

    @property
    def name(self) -> str:
        return "ide-extensions"

    @property
    def description(self) -> str:
        return "VS Code extensions required for developer ergonomics"

    def _find_code_command(self) -> str | None:
        term_program_to_command = {
            "cursor": ["cursor"],
            "vscode": ["code", "cursor"],
        }
        term_program = os.getenv("TERM_PROGRAM", "").lower()
        for cmd in term_program_to_command.get(term_program, ["cursor", "code"]):
            if shutil.which(cmd):
                return cmd
        return None

    def _list_installed_extensions(self, code_cmd: str) -> set[str]:
        result = self.run_command([code_cmd, "--list-extensions"], capture_output=True, check=False)
        if result.returncode != 0:
            warning("Unable to list VS Code extensions; assuming none are installed.")
            return set()

        return {line.strip() for line in result.stdout.splitlines() if line.strip()}

    def _is_applicable(self) -> bool:
        return bool(self._find_code_command())

    def check_installed(self) -> bool:
        if not (code_cmd := self._find_code_command()):
            return True
        installed = self._list_installed_extensions(code_cmd)
        return not (set(self._required_extensions) - installed)

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        self._non_interactive = non_interactive

        code_cmd = self._find_code_command()
        if not code_cmd:
            return
        for extension, install_path in self._required_extensions.items():
            info(f"Installing VS Code extension: {extension}")
            install_cmd = [code_cmd, "--install-extension", install_path]
            if force:
                install_cmd.append("--force")
            result = self.run_command(install_cmd, capture_output=True, check=False)

            if result.returncode == 0:
                info(f"Installed VS Code extension '{extension}'.")
                continue
            error(result.stderr)
            warning(f"Failed to install VS Code extension '{extension}'")
