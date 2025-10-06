#!/usr/bin/env python3
"""Ensure all recipe subdirectories are Python packages.

Creates __init__.py files in subdirectories that don't have them,
allowing Python to import recipes from those directories.
"""

from pathlib import Path

from metta.common.util.fs import cd_repo_root


def ensure_recipe_packages(base_dir: Path = Path("experiments/recipes")) -> list[Path]:
    """Create __init__.py files in subdirectories missing them.

    Returns:
        List of __init__.py files that were created
    """
    created = []

    for subdir in base_dir.rglob("*"):
        # Skip non-directories and __pycache__
        if not subdir.is_dir() or subdir.name == "__pycache__":
            continue

        # Skip if it's experiments/recipes itself
        if subdir == base_dir:
            continue

        init_file = subdir / "__init__.py"
        if not init_file.exists():
            # Check if directory has any .py files (potential recipes)
            py_files = list(subdir.glob("*.py"))
            if py_files:
                init_file.touch()
                created.append(init_file)
                print(f"Created: {init_file}")

    return created


if __name__ == "__main__":
    import sys

    cd_repo_root()

    created = ensure_recipe_packages()

    if created:
        print(f"\n✅ Created {len(created)} __init__.py file(s)")
        print("\nThese directories can now be imported as Python packages.")
    else:
        print("✅ All recipe subdirectories already have __init__.py files")

    sys.exit(0)
