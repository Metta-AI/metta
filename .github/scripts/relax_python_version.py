#!/usr/bin/env python3
"""
Script to temporarily relax Python version requirements during wheel building.
Changes python_requires="==3.11.7" to python_requires=">=3.11,<3.12"
"""

import re
import sys
from pathlib import Path


def patch_setup_py(filepath):
    """Patch python_requires in setup.py"""
    content = filepath.read_text()
    original_content = content

    # Match various formats of python_requires
    patterns = [
        (r'python_requires\s*=\s*["\']==3\.11\.7["\']', 'python_requires=">=3.11,<3.12"'),
        (r'python_requires\s*=\s*["\']~=3\.11\.7["\']', 'python_requires=">=3.11,<3.12"'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    if content != original_content:
        filepath.write_text(content)
        print(f"✓ Patched {filepath}")
        return True
    else:
        print(f"ℹ No changes needed in {filepath}")
        return False


def patch_pyproject_toml(filepath):
    """Patch requires-python in pyproject.toml"""
    content = filepath.read_text()
    original_content = content

    # Match various formats of requires-python
    patterns = [
        (r'requires-python\s*=\s*["\']==3\.11\.7["\']', 'requires-python = ">=3.11,<3.12"'),
        (r'requires-python\s*=\s*["\']~=3\.11\.7["\']', 'requires-python = ">=3.11,<3.12"'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    if content != original_content:
        filepath.write_text(content)
        print(f"✓ Patched {filepath}")
        return True
    else:
        print(f"ℹ No changes needed in {filepath}")
        return False


def main():
    """Main function to patch Python version requirements"""
    # Get the package directory from command line or use current directory
    package_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

    print(f"Checking for Python version requirements in: {package_dir}")

    patched = False

    # Check for setup.py
    setup_py = package_dir / "setup.py"
    if setup_py.exists():
        patched |= patch_setup_py(setup_py)
    else:
        print(f"ℹ No setup.py found in {package_dir}")

    # Check for pyproject.toml
    pyproject_toml = package_dir / "pyproject.toml"
    if pyproject_toml.exists():
        patched |= patch_pyproject_toml(pyproject_toml)
    else:
        print(f"ℹ No pyproject.toml found in {package_dir}")

    # Check for setup.cfg as well (some projects use it)
    setup_cfg = package_dir / "setup.cfg"
    if setup_cfg.exists():
        content = setup_cfg.read_text()
        original_content = content
        content = re.sub(r"python_requires\s*=\s*==3\.11\.7", "python_requires = >=3.11,<3.12", content)
        if content != original_content:
            setup_cfg.write_text(content)
            print(f"✓ Patched {setup_cfg}")
            patched = True

    if patched:
        print("\n✅ Successfully relaxed Python version requirements")
    else:
        print("\n⚠️  No Python version requirements were modified")
        print("   This might be intentional if the package doesn't have strict version requirements")

    return 0


if __name__ == "__main__":
    sys.exit(main())
