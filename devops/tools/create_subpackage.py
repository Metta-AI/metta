#!/usr/bin/env -S uv run
"""
Script to create a new subpackage in the Metta monorepo
"""

import argparse
from pathlib import Path

from metta.common.util.colorama import bold, cyan, green, magenta, red, yellow

# Assuming this script is at /devops/tools/create_subpackage.py
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def create_directory(path: Path, dry_run: bool = False) -> bool:
    """Create a directory if it doesn't exist"""
    if path.exists():
        print(yellow(f"Directory already exists: {path}"))
        return False

    if dry_run:
        print(f"{yellow('Would create directory:')} {cyan(str(path))}")
    else:
        path.mkdir(parents=True, exist_ok=True)
        print(f"{green('Created directory:')} {cyan(str(path))}")
    return True


def write_file(path: Path, content: str, dry_run: bool = False) -> bool:
    """Write content to a file"""
    if path.exists():
        print(yellow(f"File already exists: {path}"))
        return False

    if dry_run:
        print(f"{yellow('Would create file:')} {cyan(str(path))}")
        if len(content) < 200:
            print(magenta("  Content:"))
            for line in content.split("\n"):
                print(f"    {line}")
    else:
        path.write_text(content)
        print(f"{green('Created file:')} {cyan(str(path))}")
    return True


def get_makefile_content() -> str:
    """Generate Makefile content"""
    return """.PHONY: help install test clean

help:
\t@echo "Available targets:"
\t@echo "  install - Install package in editable mode"
\t@echo "  test    - Run tests with coverage"
\t@echo "  clean   - Clean build artifacts"

install:
\t@echo "Installing package in editable mode..."
\tUV_PROJECT_ENVIRONMENT=../.venv uv sync --inexact

test:
\t@echo "Running tests with coverage..."
\tUV_PROJECT_ENVIRONMENT=../.venv uv run pytest --cov=metta.{package_name} --cov-report=term-missing

clean:
\t@echo "Cleaning build artifacts..."
\t@rm -rf .venv
\t@rm -f uv.lock
\t@find . -type d -name __pycache__ -exec rm -rf {} +
\t@find . -type f -name "*.pyc" -delete
"""


def get_pyproject_content(package_name: str, author_name: str, author_email: str) -> str:
    """Generate pyproject.toml content"""
    return f"""[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "metta-{package_name}"
version = "0.1.0"
description = "Metta AI {package_name.replace("_", " ").title()} Module"
authors = [{{ name = "{author_name}", email = "{author_email}" }}]
requires-python = "==3.11.7"
license = "MIT"
readme = "README.md"
urls = {{ Homepage = "https://daveey.github.io", Repository = "https://github.com/Metta-AI/metta" }}

dependencies = [
  "pytest>=8.3.3",
  "pytest-cov>=6.1.1",
  "ruff>=0.11.13",
  "pyright>=1.1.401",
]

[tool.setuptools.packages.find]
where = ["src"]
include = ["metta.{package_name}"]

[tool.setuptools.package-data]
"metta" = ["__init__.py"]
"metta.{package_name}" = ["py.typed"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.coverage.run]
source = ["metta.{package_name}"]
"""


def get_readme_content(package_name: str) -> str:
    """Generate README.md content"""
    title = package_name.replace("_", " ").title()
    return f"""# Metta {title}

This module provides {package_name} functionality for the Metta AI system.

## Installation

From the repository root:

```bash
cd {package_name}
make install
```

## Development

### Running Tests

```bash
make test
```

### Code Quality

The project uses:
- `ruff` for linting
- `pyright` for type checking
- `pytest` for testing
- `pytest-cov` for coverage reporting

## Usage

```python
from metta.{package_name} import ...
```

## API Reference

TODO: Add API documentation

## Contributing

Please follow the project's coding standards and ensure all tests pass before submitting pull requests.
"""


def get_init_content() -> str:
    """Generate __init__.py content"""
    return '''"""Metta package namespace."""
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
'''


def get_module_init_content(package_name: str) -> str:
    """Generate module __init__.py content"""
    title = package_name.replace("_", " ").title()
    return f'''"""{title} module for Metta AI system."""

__version__ = "0.1.0"
'''


def get_test_content(package_name: str) -> str:
    """Generate basic test file content"""
    return f'''"""Tests for metta.{package_name}"""

import pytest


def test_import():
    """Test that the module can be imported"""
    import metta.{package_name}
    assert metta.{package_name}.__version__ == "0.1.0"


def test_placeholder():
    """Placeholder test - replace with actual tests"""
    # TODO: Add real tests here
    assert True
'''


def create_subpackage(package_name: str, author_name: str, author_email: str, dry_run: bool = False) -> bool:
    """Create a new subpackage with the standard structure"""

    # Validate package name
    if not package_name.replace("_", "").isalnum():
        print(red(f"Error: Package name '{package_name}' must contain only letters, numbers, and underscores"))
        return False

    # Define paths
    package_root = REPO_ROOT / package_name
    src_dir = package_root / "src"
    metta_dir = src_dir / "metta"
    module_dir = metta_dir / package_name
    tests_dir = package_root / "tests"

    # Check if package already exists
    if package_root.exists() and not dry_run:
        print(red(f"Error: Package directory '{package_name}' already exists"))
        return False

    print(bold(f"Creating subpackage: {cyan(package_name)}"))
    print(magenta("=" * 50))

    # Create directory structure
    create_directory(package_root, dry_run)
    create_directory(src_dir, dry_run)
    create_directory(metta_dir, dry_run)
    create_directory(module_dir, dry_run)
    create_directory(tests_dir, dry_run)

    # Create files
    files_created = []

    # Makefile
    makefile_content = get_makefile_content().replace("{package_name}", package_name)
    if write_file(package_root / "Makefile", makefile_content, dry_run):
        files_created.append("Makefile")

    # pyproject.toml
    pyproject_content = get_pyproject_content(package_name, author_name, author_email)
    if write_file(package_root / "pyproject.toml", pyproject_content, dry_run):
        files_created.append("pyproject.toml")

    # README.md
    readme_content = get_readme_content(package_name)
    if write_file(package_root / "README.md", readme_content, dry_run):
        files_created.append("README.md")

    # __init__.py files
    if write_file(metta_dir / "__init__.py", get_init_content(), dry_run):
        files_created.append("src/metta/__init__.py")

    if write_file(module_dir / "__init__.py", get_module_init_content(package_name), dry_run):
        files_created.append(f"src/metta/{package_name}/__init__.py")

    # py.typed marker
    if write_file(module_dir / "py.typed", "", dry_run):
        files_created.append(f"src/metta/{package_name}/py.typed")

    # Test file
    test_content = get_test_content(package_name)
    if write_file(tests_dir / f"test_{package_name}.py", test_content, dry_run):
        files_created.append(f"tests/test_{package_name}.py")

    # Test __init__.py
    if write_file(tests_dir / "__init__.py", "", dry_run):
        files_created.append("tests/__init__.py")

    print(magenta("=" * 50))

    if dry_run:
        print(yellow("Dry run complete. Run without --dry-run to create the subpackage."))
    else:
        if files_created:
            print(bold(green(f"Successfully created subpackage '{package_name}'")))
            print(green(f"Created {len(files_created)} files"))
            print("\n" + cyan("Next steps:"))
            print(f"  1. cd {package_name}")
            print("  2. make install")
            print("  3. make test")
            print(f"  4. Start developing in src/metta/{package_name}/")
        else:
            print(yellow("No new files created (all already existed)"))

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create a new subpackage in the Metta monorepo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ml_models
  %(prog)s data_processing --author "Jane Doe" --email "jane@example.com"
  %(prog)s new_feature --dry-run
        """,
    )

    parser.add_argument("name", help="Name of the new subpackage (e.g., 'ml_models', 'data_processing')")
    parser.add_argument("--author", default="David Bloomin", help="Author name (default: David Bloomin)")
    parser.add_argument("--email", default="daveey@gmail.com", help="Author email (default: daveey@gmail.com)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be created without actually creating")

    args = parser.parse_args()

    # Validate we're in the right place
    if not (REPO_ROOT / "pyproject.toml").exists():
        print(red(f"Error: Could not find repository root. Expected to find pyproject.toml at {REPO_ROOT}"))
        print(red("Make sure this script is located at /devops/tools/create_subpackage.py"))
        return 1

    success = create_subpackage(args.name, args.author, args.email, args.dry_run)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
