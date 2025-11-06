#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install

REQUIRED_NIM_VERSION = os.environ.get("COGAMES_NIM_VERSION", "2.2.6")
REQUIRED_NIMBY_VERSION = os.environ.get("COGAMES_NIMBY_VERSION", "0.1.6")
_NIMBY_CHECKED = False


def _version_tuple(version: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in version.split(".") if part.isdigit() or part.isdecimal())


def _extract_version(text: str) -> Optional[str]:
    match = re.search(r"\d+\.\d+\.\d+", text)
    return match.group(0) if match else None


def _resolve_command(binary: str) -> Optional[Path]:
    resolved = shutil.which(binary)
    if resolved:
        return Path(resolved)
    candidate = Path.home() / ".nimby" / "nim" / "bin" / binary
    if candidate.exists():
        return candidate
    return None


def _command_version(command: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            [str(command), "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    output = proc.stdout or proc.stderr or ""
    return _extract_version(output)


def _current_nim_version() -> Optional[str]:
    candidates = [
        _resolve_command("nim"),
        Path.home() / ".nimby" / "nim" / "bin" / "nim",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            version = _command_version(candidate)
            if version:
                return version
    return None


def _current_nimby() -> Tuple[Optional[Path], Optional[str]]:
    candidate = _resolve_command("nimby")
    if candidate is None:
        return None, None
    version = _command_version(candidate)
    return candidate, version


def _detect_target() -> Tuple[str, str]:
    system = platform.system()
    arch = platform.machine()

    os_map = {"Linux": "Linux", "Darwin": "macOS"}
    arch_map = {
        "x86_64": "X64",
        "amd64": "X64",
        "AMD64": "X64",
        "arm64": "ARM64",
        "aarch64": "ARM64",
    }

    os_name = os_map.get(system)
    arch_name = arch_map.get(arch)

    if not os_name or not arch_name:
        raise RuntimeError(f"Unsupported platform for Nimby bootstrap: {system} {arch}")
    return os_name, arch_name


def _download_nimby(target_dir: Path) -> Path:
    os_name, arch_name = _detect_target()
    url = f"https://github.com/treeform/nimby/releases/download/{REQUIRED_NIMBY_VERSION}/nimby-{os_name}-{arch_name}"
    destination = target_dir / "nimby"

    try:
        with urllib.request.urlopen(url) as response, open(destination, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Failed to download Nimby from {url}: {exc}") from exc

    destination.chmod(0o755)
    return destination


def _install_nim(nimby_cmd: Path) -> None:
    try:
        subprocess.run(
            [str(nimby_cmd), "use", REQUIRED_NIM_VERSION],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to install Nim {REQUIRED_NIM_VERSION} using {nimby_cmd}: {exc}") from exc


def _stage_nimby(nimby_cmd: Path) -> Path:
    bin_dir = Path.home() / ".nimby" / "nim" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    destination = bin_dir / "nimby"
    if destination.exists():
        destination.unlink()
    shutil.move(str(nimby_cmd), destination)
    destination.chmod(0o755)
    return destination


def _ensure_path_env() -> None:
    bin_dir = Path.home() / ".nimby" / "nim" / "bin"
    current_path = os.environ.get("PATH", "")
    bin_str = str(bin_dir)
    paths = current_path.split(os.pathsep) if current_path else []
    if bin_str not in paths:
        os.environ["PATH"] = os.pathsep.join([bin_str, current_path]) if current_path else bin_str


def ensure_nim_dependencies() -> None:
    global _NIMBY_CHECKED
    if _NIMBY_CHECKED:
        return

    required_nim = _version_tuple(REQUIRED_NIM_VERSION)
    required_nimby = _version_tuple(REQUIRED_NIMBY_VERSION)

    nim_version = _current_nim_version()
    nimby_cmd, nimby_version = _current_nimby()

    nim_ok = nim_version is not None and _version_tuple(nim_version) >= required_nim
    nimby_ok = nimby_cmd is not None and nimby_version is not None and _version_tuple(nimby_version) >= required_nimby

    if nim_ok and nimby_ok:
        _ensure_path_env()
        print(f"Nim {nim_version} and Nimby {nimby_version} already available.", file=sys.stdout)
        _NIMBY_CHECKED = True
        return

    if nimby_ok and nimby_cmd:
        print(f"Installing Nim {REQUIRED_NIM_VERSION} with existing Nimby at {nimby_cmd}.", file=sys.stdout)
        _install_nim(nimby_cmd)
    else:
        print(f"Bootstrapping Nimby {REQUIRED_NIMBY_VERSION} and Nim {REQUIRED_NIM_VERSION}.", file=sys.stdout)
        with tempfile.TemporaryDirectory() as tmp_dir:
            staged = _download_nimby(Path(tmp_dir))
            _install_nim(staged)
            nimby_cmd = _stage_nimby(staged)
            nimby_version = _command_version(nimby_cmd)

    nim_version = _current_nim_version()
    nimby_version = nimby_version or (_command_version(nimby_cmd) if nimby_cmd else None)

    nim_ok = nim_version is not None and _version_tuple(nim_version) >= required_nim
    nimby_ok = nimby_cmd is not None and nimby_version is not None and _version_tuple(nimby_version) >= required_nimby

    if not nim_ok or not nimby_ok:
        raise RuntimeError("Failed to provision Nim/Nimby dependencies. Inspect the logs for details.")

    _ensure_path_env()
    print(
        f"Nim {nim_version} and Nimby {nimby_version} ready at {Path.home() / '.nimby' / 'nim' / 'bin'}.",
        file=sys.stdout,
    )
    _NIMBY_CHECKED = True


class _EnsureNimbyMixin:
    def run(self) -> None:
        ensure_nim_dependencies()
        super().run()


class BuildPyCommand(_EnsureNimbyMixin, build_py):
    pass


class DevelopCommand(_EnsureNimbyMixin, develop):
    pass


class InstallCommand(_EnsureNimbyMixin, install):
    pass


setup(
    cmdclass={
        "build_py": BuildPyCommand,
        "develop": DevelopCommand,
        "install": InstallCommand,
    }
)
