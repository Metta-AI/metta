#!/usr/bin/env python3
"""
Setup script for tribal-village that builds the Nim shared library.
"""

from importlib import util
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install


def _load_build_helpers():
    """Load tribal_village_env.build without mutating sys.path."""
    project_root = Path(__file__).parent
    build_path = project_root / "tribal_village_env" / "build.py"
    spec = util.spec_from_file_location("tribal_village_env.build", build_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load build helpers from {build_path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.ensure_nim_library_current


class BuildNimLibrary:
    """Mixin class to build the Nim shared library."""

    def build_nim_library(self):
        """Build or refresh the Nim shared library using nimby + nim."""
        print("Building Nim shared library via nimby...")
        ensure_nim_library_current = _load_build_helpers()
        ensure_nim_library_current(verbose=True)


class CustomBuildPy(build_py, BuildNimLibrary):
    """Custom build_py that builds Nim library first."""

    def run(self):
        self.build_nim_library()
        super().run()


class CustomDevelop(develop, BuildNimLibrary):
    """Custom develop that builds Nim library first."""

    def run(self):
        self.build_nim_library()
        super().run()


class CustomInstall(install, BuildNimLibrary):
    """Custom install that builds Nim library first."""

    def run(self):
        self.build_nim_library()
        super().run()


if __name__ == "__main__":
    setup(
        cmdclass={
            "build_py": CustomBuildPy,
            "develop": CustomDevelop,
            "install": CustomInstall,
        }
    )
