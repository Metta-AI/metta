#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install

NIM_AGENTS_DIR = Path(__file__).parent / "src" / "cogames" / "policy" / "nim_agents"
NIMBY_LOCK = NIM_AGENTS_DIR / "nimby.lock"
REQUIRED_NIM_VERSION = os.environ.get("COGAMES_NIM_VERSION", "2.2.6")
NIMBY_VERSION = os.environ.get("COGAMES_NIMBY_VERSION", "0.1.13")


def _build_nim() -> None:
    system = platform.system()
    arch = platform.machine().lower()
    if system == "Linux":
        url = f"https://github.com/treeform/nimby/releases/download/{NIMBY_VERSION}/nimby-Linux-X64"
    elif system == "Darwin":
        suffix = "ARM64" if "arm" in arch else "X64"
        url = f"https://github.com/treeform/nimby/releases/download/{NIMBY_VERSION}/nimby-macOS-{suffix}"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    dst = Path.home() / ".nimby" / "nim" / "bin" / "nimby"
    with tempfile.TemporaryDirectory() as tmp:
        nimby = Path(tmp) / "nimby"
        urllib.request.urlretrieve(url, nimby)
        nimby.chmod(nimby.stat().st_mode | stat.S_IEXEC)
        subprocess.check_call([str(nimby), "use", REQUIRED_NIM_VERSION])

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nimby, dst)

    os.environ["PATH"] = f"{dst.parent}{os.pathsep}" + os.environ.get("PATH", "")

    # Ensure nim/nimble binaries installed by nimby are discoverable by subprocesses.
    nim_bin_dir = Path.home() / ".nimby" / "nim" / "bin"
    os.environ["PATH"] = f"{nim_bin_dir}{os.pathsep}" + os.environ.get("PATH", "")

    if NIMBY_LOCK.exists():
        subprocess.check_call(["nimby", "sync", "-g", str(NIMBY_LOCK)], cwd=NIM_AGENTS_DIR)

    result = subprocess.run(["nim", "c", "nim_agents.nim"], cwd=NIM_AGENTS_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        raise RuntimeError(f"Failed to build Nim agents: {result.returncode}")


class _EnsureNimMixin:
    def run(self, *args, **kwargs):  # type: ignore[override]
        _build_nim()
        super().run(*args, **kwargs)  # type: ignore[misc]


class BuildPyCommand(_EnsureNimMixin, build_py): ...


class DevelopCommand(_EnsureNimMixin, develop): ...


class InstallCommand(_EnsureNimMixin, install): ...


setup(
    cmdclass={
        "build_py": BuildPyCommand,
        "develop": DevelopCommand,
        "install": InstallCommand,
    }
)
