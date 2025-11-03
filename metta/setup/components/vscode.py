from __future__ import annotations

import json
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

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

    def _candidate_commands(self) -> list[str]:
        return ["code", "code-insiders", "cursor"]

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

        warning(
            "VS Code (code/cursor) CLI not detected; skipping VS Code extension installation. "
            "Enable the 'Shell Command: Install \"code\" command' option or add the CLI to PATH "
            "if you want auto-install."
        )
        return False

    def _download_extension_vsix(self, extension_id: str) -> Path | None:
        """Download a VSIX for the given extension from the public marketplace."""

        if "." not in extension_id:
            return None

        publisher, name = extension_id.split(".", 1)

        payload = json.dumps(
            {
                "filters": [
                    {
                        "criteria": [
                            {"filterType": 7, "value": extension_id},
                        ]
                    }
                ],
                "flags": 103,
            }
        ).encode()

        request = urllib.request.Request(
            "https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json;api-version=3.0-preview.1",
            },
        )

        try:
            with urllib.request.urlopen(request) as response:
                data = json.load(response)
        except urllib.error.URLError as err:
            warning(f"Unable to query VS Code marketplace for {extension_id}: {err}")
            return None

        try:
            versions = data["results"][0]["extensions"][0]["versions"]
            latest = versions[0]
            vsix_url = next(
                file["source"]
                for file in latest["files"]
                if file["assetType"] == "Microsoft.VisualStudio.Services.VSIXPackage"
            )
        except (KeyError, IndexError, StopIteration) as err:
            warning(f"Unexpected marketplace response for {extension_id}: {err}")
            return None

        tmp_dir = Path(tempfile.gettempdir())
        vsix_path = tmp_dir / f"{publisher}-{name}-{latest['version']}.vsix"

        try:
            with urllib.request.urlopen(vsix_url) as resp, vsix_path.open("wb") as handle:
                handle.write(resp.read())
        except urllib.error.URLError as err:
            warning(f"Failed to download VSIX for {extension_id}: {err}")
            return None

        return vsix_path

    def check_installed(self) -> bool:
        code_cmd = self._find_code_command()
        if not code_cmd:
            return True

        installed = self._list_installed_extensions(code_cmd)
        self._missing = [ext for ext in self._required_extensions if ext not in installed]
        return not self._missing

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        self._non_interactive = non_interactive

        code_cmd = self._find_code_command()
        if not code_cmd:
            warning(
                "VS Code CLI not found even though component is enabled. "
                "Install VS Code or Cursor and enable the shell command to let metta manage extensions."
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
            install_cmd = [code_cmd, "--install-extension", extension]
            result = self.run_command(install_cmd, capture_output=True, check=False)

            if result.returncode == 0:
                info(f"Installed VS Code extension '{extension}'.")
                continue

            stderr = (result.stderr or "").lower()
            is_cursor = Path(code_cmd).name == "cursor"
            if is_cursor and "not found" in stderr:
                vsix_path = self._download_extension_vsix(extension)
                if vsix_path:
                    info(f"Installing {extension} from VSIX bundle for Cursor: {vsix_path}")
                    vsix_result = self.run_command(
                        [code_cmd, "--install-extension", str(vsix_path)],
                        capture_output=True,
                        check=False,
                    )
                    try:
                        vsix_path.unlink()
                    except OSError:
                        pass

                    if vsix_result.returncode == 0:
                        info(f"Installed VSIX for '{extension}'.")
                        continue
                    warning(
                        f"Failed to install VSIX package for {extension} via Cursor. "
                        "See output above for details."
                    )

            warning(
                f"Failed to install VS Code extension '{extension}'. Install manually with: "
                f"{code_cmd} --install-extension {extension}"
            )

        # Reset missing list after attempt; subsequent checks will refresh it
        self._missing = []
