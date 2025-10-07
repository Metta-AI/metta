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
METTASCOPE_DIR = PROJECT_ROOT / "nim" / "mettascope"
PYTHON_PACKAGE_DIR = PROJECT_ROOT / "python" / "src" / "mettagrid"
METTASCOPE_PACKAGE_DIR = PYTHON_PACKAGE_DIR / "nim" / "mettascope"


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

    # Align Bazel's registered Python toolchain with the active interpreter.
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    env = os.environ.copy()
    env.setdefault("METTAGRID_BAZEL_PYTHON_VERSION", py_version)

    # Provide a writable output root for environments with restricted /var/tmp access.
    output_user_root = env.get(
        "METTAGRID_BAZEL_OUTPUT_ROOT",
        str(PROJECT_ROOT / ".bazel_output"),
    )

    # Ensure the output root exists before invoking Bazel.
    Path(output_user_root).mkdir(parents=True, exist_ok=True)

    # Build the Python extension with auto-detected parallelism
    cmd = [
        "bazel",
        "--batch",
        f"--output_user_root={output_user_root}",
        "build",
        f"--config={config}",
        "--jobs=auto",
        "--verbose_failures",
        "//cpp:mettagrid_c",  # Build from new cpp location
    ]

    print(f"Running Bazel build: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print("Bazel build failed. STDERR:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        print("Bazel build STDOUT:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        raise RuntimeError("Bazel build failed")

    # Copy the built extension to the package directory
    bazel_bin = PROJECT_ROOT / "bazel-bin"
    # Try both old and new locations for backward compatibility
    src_dirs = [
        PROJECT_ROOT / "python/src/mettagrid",  # New location
        PROJECT_ROOT / "src/mettagrid",  # Old location (compatibility)
    ]

    # Find the built extension file
    # Bazel outputs the extension at bazel-bin/cpp/mettagrid_c.so or bazel-bin/mettagrid_c.so
    extension_patterns = [
        "cpp/mettagrid_c.so",
        "cpp/mettagrid_c.pyd",
        "cpp/mettagrid_c.dylib",  # New location
        "mettagrid_c.so",
        "mettagrid_c.pyd",
        "mettagrid_c.dylib",  # Old location
    ]
    extension_file = None
    for pattern in extension_patterns:
        file_path = bazel_bin / pattern
        if file_path.exists():
            extension_file = file_path
            break
    if not extension_file:
        raise RuntimeError("mettagrid_c.{so,pyd,dylib} not found in bazel-bin/cpp/ or bazel-bin/")

    # Copy to all source directories that exist
    for src_dir in src_dirs:
        if src_dir.parent.exists():
            # Ensure destination directory exists
            src_dir.mkdir(parents=True, exist_ok=True)
            # Copy the extension to the source directory
            dest = src_dir / extension_file.name
            # Remove existing file if it exists (it might be read-only from previous build)
            if dest.exists():
                dest.unlink()
            shutil.copy2(extension_file, dest)
            print(f"Copied {extension_file} to {dest}")


def _nim_artifacts_up_to_date() -> bool:
    """Check whether Nim outputs are still current."""

    force_rebuild = os.environ.get("METTAGRID_FORCE_NIM_BUILD", "").lower() in {"1", "true", "yes"}
    if force_rebuild:
        return False

    generated_dir = METTASCOPE_DIR / "bindings" / "generated"
    if not generated_dir.exists():
        return False

    existing_outputs = {
        generated_dir / name
        for name in (
            "mettascope2.py",
            "libmettascope2.dylib",
            "libmettascope2.so",
            "libmettascope2.dll",
        )
        if (generated_dir / name).exists()
    }
    if not existing_outputs:
        return False

    source_files = [path for pattern in ("*.nim", "*.nims") for path in METTASCOPE_DIR.rglob(pattern) if path.is_file()]
    if not source_files:
        return False

    latest_source_mtime = max(path.stat().st_mtime for path in source_files)
    oldest_output_mtime = min(path.stat().st_mtime for path in existing_outputs)

    return oldest_output_mtime >= latest_source_mtime


def _sync_mettascope_package_data() -> None:
    """Ensure Nim artifacts are vendored inside the Python package."""

    destination_root = METTASCOPE_PACKAGE_DIR
    destination_root.parent.mkdir(parents=True, exist_ok=True)

    if destination_root.exists():
        shutil.rmtree(destination_root)

    shutil.copytree(
        METTASCOPE_DIR,
        destination_root,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "nimbledeps", "dist", "build"),
    )


def _run_mettascope_build() -> None:
    """Build Nim artifacts when cache misses."""

    if _nim_artifacts_up_to_date():
        print("Skipping Nim build; artifacts up to date.")
        _sync_mettascope_package_data()
        return

    # Check if nim and nimble are available
    if shutil.which("nim") is None:
        print("Warning: Nim compiler not found. Skipping mettascope build.")
        print("To build mettascope, install Nim: https://nim-lang.org/install.html")
        raise RuntimeError("Nim compiler not found")

    if shutil.which("nimble") is None:
        print("Warning: Nimble package manager not found. Skipping mettascope build.")
        print("To build mettascope, install Nim: https://nim-lang.org/install.html")
        raise RuntimeError("Nimble package manager not found")

    print(f"Building mettascope from {METTASCOPE_DIR}")

    # Run the build script
    for cmd in ["update", "install", "bindings"]:
        result = subprocess.run(["nimble", cmd, "-y"], cwd=METTASCOPE_DIR, capture_output=True, text=True)
        print(result.stderr, file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        if result.returncode != 0:
            print(f"Warning: Mettascope build failed. {cmd} failed. STDERR:", file=sys.stderr)
            print(f"Mettascope build {cmd} STDOUT:", file=sys.stderr)
            raise RuntimeError("Mettascope build failed")
    print("Successfully built mettascope")
    _sync_mettascope_package_data()


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a wheel, compiling the C++ extension with Bazel first, then mettascope."""
    _run_bazel_build()
    _run_mettascope_build()
    # Ensure wheel is tagged as non-pure (platform-specific) since we bundle a native extension
    # Setuptools/wheel derive purity from Distribution.has_ext_modules(). Monkeypatch to force True.
    original_has_ext_modules = Distribution.has_ext_modules
    try:
        Distribution.has_ext_modules = lambda self: True  # type: ignore[assignment]
        return _build_wheel(wheel_directory, config_settings, metadata_directory)
    finally:
        Distribution.has_ext_modules = original_has_ext_modules


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build an editable install, compiling the C++ extension with Bazel first, then mettascope."""
    _run_bazel_build()
    _run_mettascope_build()
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
