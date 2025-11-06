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
from typing import Optional, Tuple

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install

REQUIRED_NIM_VERSION = os.environ.get("COGAMES_NIM_VERSION", "2.2.6")
REQUIRED_NIMBY_VERSION = os.environ.get("COGAMES_NIMBY_VERSION", "0.1.6")
NIMBY_HOME = Path.home() / ".nimby" / "nim" / "bin"
_CHECKED = False


def _version_tuple(raw: Optional[str]) -> Tuple[int, ...]:
    match = re.search(r"\d+\.\d+\.\d+", raw or "")
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()


def _resolve(binary: str) -> Optional[Path]:
    for candidate in (shutil.which(binary), NIMBY_HOME / binary):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def _command_output(binary: Path, *args: str) -> str:
    try:
        result = subprocess.run([str(binary), *args], capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return result.stdout or result.stderr or ""


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


def _ensure_path() -> None:
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(NIMBY_HOME) not in path_parts:
        path_parts = [str(NIMBY_HOME)] + [p for p in path_parts if p]
        os.environ["PATH"] = os.pathsep.join(path_parts)


def ensure_nim_dependencies() -> None:
    global _CHECKED
    if _CHECKED:
        return

    nim_cmd = _resolve("nim")
    nimby_cmd = _resolve("nimby") or _bootstrap_nimby()

    nim_version = _version_tuple(_command_output(nim_cmd, "--version") if nim_cmd else None)
    nimby_version = _version_tuple(_command_output(nimby_cmd, "--version"))

    if nim_version < _version_tuple(REQUIRED_NIM_VERSION):
        subprocess.run([str(nimby_cmd), "use", REQUIRED_NIM_VERSION], check=True)
        nim_cmd = _resolve("nim")

    if nimby_version < _version_tuple(REQUIRED_NIMBY_VERSION):
        nimby_cmd = _bootstrap_nimby()

    _ensure_path()
    _CHECKED = True


class EnsureNimbyMixin:
    def run(self) -> None:  # type: ignore[override]
        ensure_nim_dependencies()
        super().run()


class BuildPyCommand(EnsureNimbyMixin, build_py): ...


class DevelopCommand(EnsureNimbyMixin, develop): ...


class InstallCommand(EnsureNimbyMixin, install): ...


setup(
    cmdclass={
        "build_py": BuildPyCommand,
        "develop": DevelopCommand,
        "install": InstallCommand,
    }
)
