# Custom build backend for building mettagrid with Bazel
# This backend compiles the C++ extension using Bazel during package installation

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from setuptools.build_meta import build_editable as _build_editable
from setuptools.build_meta import build_wheel as _build_wheel


def _run_bazel_build():
    """Run Bazel build to compile the C++ extension."""
    # Check if bazel is available
    if shutil.which("bazel") is None:
        raise RuntimeError(
            "Bazel is required to build metta-mettagrid. "
            "Please install Bazel: https://bazel.build/install"
        )

    # Determine build configuration from environment
    debug = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
    config = "dbg" if debug else "opt"

    # Build the Python extension
    cmd = [
        "bazel",
        "build",
        f"--config={config}",
        "//:mettagrid_c",
    ]

    print(f"Running Bazel build: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Bazel build failed:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError("Bazel build failed")

    # Copy the built extension to the package directory
    bazel_bin = Path("bazel-bin")
    src_dir = Path("src/metta/mettagrid")

    # Find the built extension file
    # Bazel outputs the extension directly to bazel-bin/mettagrid_c.so
    extension_patterns = ["mettagrid_c.so", "mettagrid_c.pyd", "mettagrid_c.dylib"]
    extension_file = None

    for pattern in extension_patterns:
        file_path = bazel_bin / pattern
        if file_path.exists():
            extension_file = file_path
            break

    if extension_file:
        # Ensure destination directory exists
        src_dir.mkdir(parents=True, exist_ok=True)
        # Copy the extension to the source directory
        dest = src_dir / extension_file.name
        # Remove existing file if it exists (it might be read-only from previous build)
        if dest.exists():
            dest.unlink()
        shutil.copy2(extension_file, dest)
        print(f"Copied {extension_file} to {dest}")
    else:
        # If no pre-built extension found, we'll let setuptools handle it
        print("No pre-built extension found, continuing with standard build")


def build_wheel(wheel_directory, config_settings = None, metadata_directory = None):
    """Build a wheel, compiling the C++ extension with Bazel first."""
    _run_bazel_build()
    return _build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings = None, metadata_directory = None):
    """Build an editable install, compiling the C++ extension with Bazel first."""
    _run_bazel_build()
    return _build_editable(wheel_directory, config_settings, metadata_directory)


# Re-export other required functions from setuptools
from setuptools.build_meta import (
    get_requires_for_build_wheel,
    get_requires_for_build_editable,
    prepare_metadata_for_build_wheel,
    prepare_metadata_for_build_editable,
)

__all__ = [
    "build_wheel",
    "build_editable",
    "get_requires_for_build_wheel",
    "get_requires_for_build_editable",
    "prepare_metadata_for_build_wheel",
    "prepare_metadata_for_build_editable",
]
