#!/usr/bin/env python3
from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install

REQUIRED_NIM_VERSION = os.environ.get("COGAMES_NIM_VERSION", "2.2.6")
NIMBY_VERSION = os.environ.get("COGAMES_NIMBY_VERSION", "0.1.6")


def ensure_nim_dependencies() -> None:
    system = platform.system()
    arch = platform.machine().lower()
    if system == "Linux":
        url = f"https://github.com/treeform/nimby/releases/download/{NIMBY_VERSION}/nimby-Linux-X64"
    elif system == "Darwin":
        suffix = "ARM64" if "arm" in arch else "X64"
        url = f"https://github.com/treeform/nimby/releases/download/{NIMBY_VERSION}/nimby-macOS-{suffix}"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    with tempfile.TemporaryDirectory() as tmp:
        nimby = Path(tmp) / "nimby"
        urllib.request.urlretrieve(url, nimby)
        nimby.chmod(nimby.stat().st_mode | stat.S_IEXEC)
        subprocess.check_call([str(nimby), "use", REQUIRED_NIM_VERSION])

        dst = Path.home() / ".nimby" / "nim" / "bin" / "nimby"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nimby, dst)

    os.environ["PATH"] = f"{dst.parent}{os.pathsep}" + os.environ.get("PATH", "")


class _EnsureNimMixin:
    def run(self, *args, **kwargs):  # type: ignore[override]
        ensure_nim_dependencies()
        super().run(*args, **kwargs)


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
