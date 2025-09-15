# Custom build backend for building mettagrid with Bazel
# This backend compiles the C++ extension using Bazel during package installation

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools.build_meta import (
    build_editable as _build_editable,
)
from setuptools.build_meta import (
    build_sdist as _build_sdist,
)
from setuptools.build_meta import (
    build_wheel as _build_wheel,
)
from setuptools.build_meta import (
    get_requires_for_build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel,
)
from setuptools.dist import Distribution

PROJECT_ROOT = Path(__file__).resolve().parent


def _run_bazel_build() -> None:
    """Run Bazel build to compile the C++ extension."""
    # Check if bazel is available
    if shutil.which("bazel") is None:
        raise RuntimeError("Bazel is required to build mettagrid. Please install Bazel: https://bazel.build/install")

    # Determine build configuration from environment
    debug = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")

    # Check if running in CI environment (GitHub Actions sets CI=true)
    is_ci = os.environ.get("CI", "").lower() == "true" or os.environ.get("GITHUB_ACTIONS", "") == "true"

    if is_ci:
        # Use CI configuration to avoid root user issues with hermetic Python
        config = "ci"
    else:
        config = "dbg" if debug else "opt"

    # Build the Python extension
    cmd = [
        "bazel",
        "build",
        f"--config={config}",
        "--verbose_failures",
        "//:mettagrid_c",
    ]

    print(f"Running Bazel build: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        print("Bazel build failed. STDERR:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        print("Bazel build STDOUT:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        raise RuntimeError("Bazel build failed")

    # Copy the built extension to the package directory
    bazel_bin = PROJECT_ROOT / "bazel-bin"
    src_dir = PROJECT_ROOT / "src/metta/mettagrid"

    # Find the built extension file
    # Bazel outputs the extension directly to bazel-bin/mettagrid_c.so
    extension_patterns = ["mettagrid_c.so", "mettagrid_c.pyd", "mettagrid_c.dylib"]
    extension_file = None
    for pattern in extension_patterns:
        file_path = bazel_bin / pattern
        if file_path.exists():
            extension_file = file_path
            break
    if not extension_file:
        raise RuntimeError("mettagrid_c.{so,pyd,dylib} not found")

    # Ensure destination directory exists
    src_dir.mkdir(parents=True, exist_ok=True)
    # Copy the extension to the source directory
    dest = src_dir / extension_file.name
    # Remove existing file if it exists (it might be read-only from previous build)
    if dest.exists():
        dest.unlink()
    shutil.copy2(extension_file, dest)
    print(f"Copied {extension_file} to {dest}")


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a wheel, compiling the C++ extension with Bazel first."""
    _run_bazel_build()
    # Ensure wheel is tagged as non-pure (platform-specific) since we bundle a native extension
    # Setuptools/wheel derive purity from Distribution.has_ext_modules(). Monkeypatch to force True.
    original_has_ext_modules = Distribution.has_ext_modules
    try:
        Distribution.has_ext_modules = lambda self: True  # type: ignore[assignment]
        return _build_wheel(wheel_directory, config_settings, metadata_directory)
    finally:
        Distribution.has_ext_modules = original_has_ext_modules


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build an editable install, compiling the C++ extension with Bazel first."""
    _run_bazel_build()
    return _build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    """Build a source distribution without compiling the extension."""
    return _build_sdist(sdist_directory, config_settings)


__all__ = [
    "build_wheel",
    "build_editable",
    "build_sdist",
    "get_requires_for_build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "prepare_metadata_for_build_wheel",
    "prepare_metadata_for_build_editable",
]
