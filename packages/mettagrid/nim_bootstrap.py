"""Bootstrap Nimby + Nim for Bazel builds, mirroring treeform's recipe."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

REQUIRED_NIM_VERSION = os.environ.get("METTAGRID_NIM_VERSION", "2.2.6")
REQUIRED_NIMBY_VERSION = os.environ.get("METTAGRID_NIMBY_VERSION", "0.1.6")
NIMBY_HOME = Path.home() / ".nimby" / "nim" / "bin"
NIM_BIN = NIMBY_HOME / "nim"
NIMBY_BIN = NIMBY_HOME / "nimby"
_BOOTSTRAPPED = False


def ensure_nim_dependencies() -> None:
    """Download Nimby, run `nimby use`, and put the binaries on PATH."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    os_name = {"Linux": "Linux", "Darwin": "macOS"}.get(platform.system())
    arch_name = {"x86_64": "X64", "amd64": "X64", "arm64": "ARM64", "aarch64": "ARM64"}.get(platform.machine())
    if not os_name or not arch_name:
        raise RuntimeError("Unsupported platform for Nimby download")

    url = f"https://github.com/treeform/nimby/releases/download/{REQUIRED_NIMBY_VERSION}/nimby-{os_name}-{arch_name}"

    with tempfile.TemporaryDirectory() as tmp:
        nimby_path = Path(tmp) / "nimby"
        with urllib.request.urlopen(url, timeout=30) as resp, open(nimby_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        nimby_path.chmod(0o755)

        subprocess.run([str(nimby_path), "use", REQUIRED_NIM_VERSION], check=True)

        NIMBY_HOME.mkdir(parents=True, exist_ok=True)
        if NIMBY_BIN.exists():
            NIMBY_BIN.unlink()
        shutil.move(str(nimby_path), NIMBY_BIN)

    path = os.environ.get("PATH", "")
    if str(NIMBY_HOME) not in path.split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join([str(NIMBY_HOME), path]) if path else str(NIMBY_HOME)

    _BOOTSTRAPPED = True


__all__ = ["ensure_nim_dependencies", "NIM_BIN", "NIMBY_BIN"]
