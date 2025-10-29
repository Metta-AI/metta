#!/usr/bin/env python3
"""
Setup script for tribal-village that builds the Nim shared library.
"""

import os
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install


class BuildNimLibrary:
    """Mixin class to build the Nim shared library."""

    def build_nim_library(self):
        """Build the Nim shared library using build_lib.sh"""
        print("Building Nim shared library...")

        # Get the project root directory
        project_root = Path(__file__).parent
        build_script = project_root / "build_lib.sh"

        # If a prebuilt library is already present, skip rebuilding.
        prebuilt = None
        for ext in (".so", ".dylib", ".dll"):
            candidate = project_root / f"libtribal_village{ext}"
            if candidate.exists():
                prebuilt = candidate
                break

        if prebuilt is not None:
            print(f"Using existing Nim library at {prebuilt}")
        elif build_script.exists():
            # Run custom build script if provided
            result = subprocess.run(
                ["bash", str(build_script)],
                cwd=project_root,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to build Nim library: {result.stderr}")
        else:
            # Fall back to Nimble build if script is absent
            result = subprocess.run(
                ["nimble", "buildLib"],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Failed to build Nim library with Nimble. "
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                )

        # Copy the built library to the Python package directory
        lib_file = prebuilt
        if lib_file is None:
            for ext in (".so", ".dylib", ".dll"):
                candidate = project_root / f"libtribal_village{ext}"
                if candidate.exists():
                    lib_file = candidate
                    break

        if lib_file is None:
            raise RuntimeError("Nim library was not created by build step")

        package_dir = project_root / "tribal_village_env"
        target_name = "libtribal_village.so"
        if lib_file.suffix == ".dylib":
            target_name = "libtribal_village.dylib"
        elif lib_file.suffix == ".dll":
            target_name = "libtribal_village.dll"

        shutil.copy2(lib_file, package_dir / target_name)
        print(f"Copied {lib_file} to {package_dir / target_name}")


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
            'build_py': CustomBuildPy,
            'develop': CustomDevelop,
            'install': CustomInstall,
        }
    )
