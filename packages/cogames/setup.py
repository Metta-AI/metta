#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, TypeVar

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install

REQUIRED_NIM_VERSION = os.environ.get("COGAMES_NIM_VERSION", "2.2.6")
REQUIRED_NIMBY_VERSION = os.environ.get("COGAMES_NIMBY_VERSION", "0.1.6")
NIMBY_HOME = Path.home() / ".nimby" / "nim" / "bin"
_CHECKED = False


def _version_parts(raw: Optional[str]) -> Tuple[int, ...]:
    match = re.search(r"\d+\.\d+\.\d+", raw or "")
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()


def _resolve(binary: str) -> Optional[Path]:
    for candidate in (shutil.which(binary), NIMBY_HOME / binary):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def _bootstrap_nimby() -> Path:
    system = {"Linux": "Linux", "Darwin": "macOS"}.get(platform.system())
    arch = {"x86_64": "X64", "amd64": "X64", "arm64": "ARM64", "aarch64": "ARM64"}.get(platform.machine())
    if not system or not arch:
        raise RuntimeError("Unsupported platform for Nimby download")

    url = f"https://github.com/treeform/nimby/releases/download/{REQUIRED_NIMBY_VERSION}/nimby-{system}-{arch}"
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "nimby"
        with urllib.request.urlopen(url, timeout=30) as resp, open(target, "wb") as out:
            shutil.copyfileobj(resp, out)
        target.chmod(0o755)
        NIMBY_HOME.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), NIMBY_HOME / "nimby")
    return NIMBY_HOME / "nimby"


def ensure_nim_dependencies() -> None:
    global _CHECKED
    if _CHECKED:
        return

    nim_cmd = _resolve("nim")
    nimby_cmd = _resolve("nimby") or _bootstrap_nimby()

    nim_version = _version_parts(
        subprocess.run([str(nim_cmd), "--version"], capture_output=True, text=True).stdout if nim_cmd else None
    )
    nimby_version = _version_parts(subprocess.run([str(nimby_cmd), "--version"], capture_output=True, text=True).stdout)

    if nim_version < _version_parts(REQUIRED_NIM_VERSION):
        subprocess.run([str(nimby_cmd), "use", REQUIRED_NIM_VERSION], check=True)

    if nimby_version < _version_parts(REQUIRED_NIMBY_VERSION):
        nimby_cmd = _bootstrap_nimby()
        subprocess.run([str(nimby_cmd), "use", REQUIRED_NIM_VERSION], check=True)

    path = os.environ.get("PATH", "")
    if str(NIMBY_HOME) not in path.split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join([str(NIMBY_HOME), path]) if path else str(NIMBY_HOME)

    _CHECKED = True


CmdClass = TypeVar("CmdClass", bound=type)


def _wrap(cmd_class: CmdClass) -> CmdClass:
    class Wrapped(cmd_class):  # type: ignore[misc,valid-type]
        def run(self, *args, **kwargs):  # type: ignore[override]
            ensure_nim_dependencies()
            super().run(*args, **kwargs)

    return Wrapped


setup(
    cmdclass={
        "build_py": _wrap(build_py),
        "develop": _wrap(develop),
        "install": _wrap(install),
    }
)
