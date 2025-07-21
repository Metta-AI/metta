#!/usr/bin/env python3
"""Quick verification script to check PufferLib integration setup."""

import sys
import subprocess
from pathlib import Path


def check_command(cmd, name):
    """Check if a command is available."""
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"✓ {name} is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"✗ {name} is not available")
        return False


def check_file(path, name):
    """Check if a file exists and is executable."""
    if path.exists():
        print(f"✓ {name} exists")
        if path.is_file() and path.stat().st_mode & 0o111:
            print(f"  ✓ {name} is executable")
            return True
        else:
            print(f"  ✗ {name} is not executable")
            return False
    else:
        print(f"✗ {name} does not exist")
        return False


def main():
    """Run verification checks."""
    print("=== PufferLib Integration Test Setup Verification ===\n")
    
    all_good = True
    
    # Check required commands
    print("Checking required commands:")
    all_good &= check_command(["python", "--version"], "Python")
    all_good &= check_command(["git", "--version"], "Git")
    all_good &= check_command(["cmake", "--version"], "CMake")
    all_good &= check_command(["uv", "--version"], "uv")
    all_good &= check_command(["docker", "--version"], "Docker (optional)")
    
    print("\nChecking test files:")
    test_dir = Path(__file__).parent
    all_good &= check_file(test_dir / "test_pufferlib_fresh_install.sh", "Fresh install test script")
    all_good &= check_file(test_dir / "test_pufferlib_docker.sh", "Docker test script")
    
    print("\nChecking Python test file:")
    test_file = test_dir / "test_pufferlib_integration.py"
    if test_file.exists():
        print("✓ test_pufferlib_integration.py exists")
        # Try to import it
        try:
            sys.path.insert(0, str(test_dir))
            import test_pufferlib_integration
            print("  ✓ Can be imported")
        except ImportError as e:
            print(f"  ✗ Cannot be imported: {e}")
            all_good = False
    else:
        print("✗ test_pufferlib_integration.py does not exist")
        all_good = False
    
    print("\nChecking documentation:")
    readme = test_dir / "README.md"
    if readme.exists():
        print("✓ README.md exists")
        lines = readme.read_text().splitlines()
        print(f"  {len(lines)} lines of documentation")
    else:
        print("✗ README.md does not exist")
        all_good = False
    
    print("\nChecking CI configuration:")
    ci_file = Path(__file__).parent.parent.parent / ".github/workflows/pufferlib-integration.yml"
    if ci_file.exists():
        print("✓ CI workflow exists")
    else:
        print("✗ CI workflow does not exist")
        all_good = False
    
    print("\n" + "="*50)
    if all_good:
        print("✅ All checks passed! Setup is ready.")
        return 0
    else:
        print("❌ Some checks failed. Please install missing dependencies.")
        print("\nTo install uv:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("\nTo install other dependencies:")
        print("  # Ubuntu/Debian: sudo apt-get install git build-essential cmake")
        print("  # macOS: brew install cmake")
        return 1


if __name__ == "__main__":
    sys.exit(main())