"""Bootstrap Nimby + Nim for Bazel builds without extra bells and whistles."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REQUIRED_NIM_VERSION = os.environ.get("METTAGRID_NIM_VERSION", "2.2.6")
REQUIRED_NIMBY_VERSION = os.environ.get("METTAGRID_NIMBY_VERSION", "0.1.6")
NIM_HOME = Path.home() / ".nimby" / "nim"
NIM_BIN_DIR = NIM_HOME / "bin"
INSTALL_BIN_DIR = Path.home() / ".local" / "bin"
NIMBY_BIN = INSTALL_BIN_DIR / "nimby"
NIM_BIN = NIM_BIN_DIR / "nim"
_BOOTSTRAPPED = False


def _download_and_stage() -> Path:
    os_name = {"Linux": "Linux", "Darwin": "macOS"}.get(platform.system())
    arch_name = {"x86_64": "X64", "amd64": "X64", "arm64": "ARM64", "aarch64": "ARM64"}.get(platform.machine())
    if not os_name or not arch_name:
        raise RuntimeError("Unsupported platform for Nimby download")

    url = f"https://github.com/treeform/nimby/releases/download/{REQUIRED_NIMBY_VERSION}/nimby-{os_name}-{arch_name}"
    NIM_BIN_DIR.mkdir(parents=True, exist_ok=True)
    INSTALL_BIN_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "nimby"
        try:
            with urllib.request.urlopen(url, timeout=30) as response, open(target, "wb") as handle:
                shutil.copyfileobj(response, handle)
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Failed to download Nimby from {url}: {exc}") from exc
        target.chmod(0o755)
        subprocess.run([str(target), "use", REQUIRED_NIM_VERSION], check=True)
        shutil.copy2(target, NIMBY_BIN)

    return NIMBY_BIN


def ensure_nim_dependencies() -> None:
    """Always download Nimby, install Nim, and put binaries on PATH."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    nimby = _download_and_stage()

    path = os.environ.get("PATH", "")
    new_paths = [str(NIM_BIN_DIR), str(INSTALL_BIN_DIR)]
    current = path.split(os.pathsep) if path else []
    for bin_dir in new_paths:
        if bin_dir and bin_dir not in current:
            current.insert(0, bin_dir)
    os.environ["PATH"] = os.pathsep.join(current)

    _BOOTSTRAPPED = True


__all__ = ["ensure_nim_dependencies", "NIMBY_BIN", "NIM_BIN"]
