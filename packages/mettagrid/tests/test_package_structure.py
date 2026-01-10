"""Tests to ensure mettagrid package structure remains ship-ready.

These tests verify that all packages have proper __init__.py files and are
discoverable, which is critical for Bazel binary wheel builds.
"""

import importlib
import pkgutil
from pathlib import Path

import mettagrid


def test_all_package_directories_have_init_py():
    """Verify all Python package directories have __init__.py files.

    Directories without __init__.py are namespace packages that may be excluded
    from binary wheels built by Bazel.
    """
    mettagrid_root = Path(mettagrid.__file__).parent

    # Find all directories that contain .py files (potential packages)
    package_dirs = set()
    for py_file in mettagrid_root.rglob("*.py"):
        # Skip __pycache__ and test files
        if "__pycache__" in py_file.parts:
            continue

        # Add the directory containing this .py file
        package_dirs.add(py_file.parent)

    # Verify each package directory has an __init__.py
    missing_init = []
    for pkg_dir in sorted(package_dirs):
        init_file = pkg_dir / "__init__.py"
        if not init_file.exists():
            # Make path relative to mettagrid root for clearer error messages
            rel_path = pkg_dir.relative_to(mettagrid_root)
            missing_init.append(str(rel_path))

    assert not missing_init, (
        f"The following package directories are missing __init__.py files:\n"
        f"{chr(10).join(f'  - mettagrid/{p}/' for p in missing_init)}\n\n"
        f"Directories without __init__.py are namespace packages that may be "
        f"excluded from binary wheels. Add an empty __init__.py to each directory."
    )


def test_all_packages_are_discoverable():
    """Verify all mettagrid packages are discoverable via pkgutil.

    This ensures packages will be properly included in distribution builds.
    """
    # Expected top-level packages (directories with __init__.py under mettagrid/)
    expected_packages = {
        "mettagrid.builder",
        "mettagrid.config",
        "mettagrid.envs",
        "mettagrid.map_builder",
        "mettagrid.mapgen",
        "mettagrid.policy",
        "mettagrid.profiling",
        "mettagrid.renderer",
        "mettagrid.simulator",
        "mettagrid.test_support",
        "mettagrid.util",
    }

    # Discover all packages
    discovered = set()
    for importer, modname, ispkg in pkgutil.walk_packages(
        mettagrid.__path__, mettagrid.__name__ + "."
    ):
        if ispkg:
            # Extract top-level package name (e.g., mettagrid.mapgen.scenes -> mettagrid.mapgen)
            parts = modname.split(".")
            if len(parts) >= 2:
                top_level = f"{parts[0]}.{parts[1]}"
                discovered.add(top_level)

    missing = expected_packages - discovered
    assert not missing, (
        f"Expected packages not discovered via pkgutil.walk_packages():\n"
        f"{chr(10).join(f'  - {p}' for p in sorted(missing))}\n\n"
        f"This likely means these directories are missing __init__.py files."
    )


def test_all_packages_importable():
    """Verify all major mettagrid packages can be imported without errors."""
    packages_to_test = [
        "mettagrid.builder",
        "mettagrid.config",
        "mettagrid.envs",
        "mettagrid.map_builder",
        "mettagrid.mapgen",
        "mettagrid.policy",
        "mettagrid.profiling",
        "mettagrid.renderer",
        "mettagrid.simulator",
        "mettagrid.util",
    ]

    import_errors = []
    for package_name in packages_to_test:
        try:
            importlib.import_module(package_name)
        except Exception as e:
            import_errors.append(f"{package_name}: {e}")

    assert not import_errors, (
        f"Failed to import the following packages:\n"
        f"{chr(10).join(f'  - {err}' for err in import_errors)}"
    )


def test_no_circular_imports_with_config():
    """Verify policy package doesn't have circular imports with config.

    This was a historical issue that was fixed. This test ensures it stays fixed.
    """
    # This import chain previously caused circular imports:
    # config -> simulator -> util.file -> policy -> config
    from mettagrid.config.mettagrid_c_config import convert_to_cpp_game_config
    from mettagrid.policy.policy import MultiAgentPolicy

    assert convert_to_cpp_game_config is not None
    assert MultiAgentPolicy is not None


def test_critical_packages_have_proper_init():
    """Verify critical packages that were previously namespace packages now have __init__.py."""
    mettagrid_root = Path(mettagrid.__file__).parent

    # These directories previously lacked __init__.py and would be excluded from wheels
    critical_packages = [
        "builder",
        "envs",
        "mapgen",
        "profiling",
        "renderer",
        "util",
    ]

    missing = []
    for pkg in critical_packages:
        init_file = mettagrid_root / pkg / "__init__.py"
        if not init_file.exists():
            missing.append(pkg)

    assert not missing, (
        f"Critical packages missing __init__.py (will be excluded from binary wheels):\n"
        f"{chr(10).join(f'  - mettagrid/{p}/' for p in missing)}"
    )
