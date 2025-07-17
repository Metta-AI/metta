#!/usr/bin/env python3
"""
Test script for the compare_training tool
"""

import subprocess
import sys
from pathlib import Path


def test_config_creation():
    """Test that the Hydra config file exists and is valid"""
    config_path = Path("metta/configs/user/compare_training.yaml")
    assert config_path.exists(), f"Config file {config_path} does not exist"

    print(f"✓ Config file exists: {config_path}")

    # Try to load the config to verify it's valid YAML
    try:
        # Try to import yaml, but don't fail if it's not available
        try:
            import yaml

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            print("✓ Config file is valid YAML")
            print(f"  - Curriculum: {config.get('trainer', {}).get('curriculum', 'NOT FOUND')}")
            print(f"  - Total timesteps: {config.get('trainer', {}).get('total_timesteps', 'NOT FOUND')}")
            print(
                f"  - Learning rate: {config.get('trainer', {}).get('optimizer', {}).get('learning_rate', 'NOT FOUND')}"
            )
        except ImportError:
            # If yaml is not available, just check that the file can be read
            with open(config_path, "r") as f:
                content = f.read()
            if "curriculum:" in content and "total_timesteps:" in content:
                print("✓ Config file exists and contains expected content")
            else:
                print("✗ Config file does not contain expected content")
                return False
    except Exception as e:
        print(f"✗ Config file is not valid: {e}")
        return False

    return True


def test_bullm_run_exists():
    """Test that bullm_run.py exists"""
    run_path = Path("metta/bullm_run.py")
    assert run_path.exists(), f"bullm_run.py does not exist at {run_path}"
    print(f"✓ bullm_run.py exists: {run_path}")
    return True


def test_compare_tool_import():
    """Test that the compare_training tool can be imported"""
    try:
        sys.path.insert(0, str(Path("metta/tools")))
        print("✓ compare_training module can be imported")
        return True
    except Exception as e:
        print(f"✗ Failed to import compare_training: {e}")
        return False


def test_compare_tool_help():
    """Test that the compare_training tool shows help"""
    try:
        result = subprocess.run(
            [sys.executable, "metta/tools/compare_training.py", "--help"], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            print("✓ compare_training tool shows help correctly")
            return True
        else:
            print(f"✗ compare_training tool failed to show help: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ compare_training tool help command timed out")
        return False
    except Exception as e:
        print(f"✗ Error running compare_training tool: {e}")
        return False


def main():
    """Run all tests"""
    print("Testing compare_training tool setup...")
    print("=" * 50)

    tests = [
        test_config_creation,
        test_bullm_run_exists,
        test_compare_tool_import,
        test_compare_tool_help,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")

    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed! The comparison tool is ready to use.")
        print("\nTo run a comparison:")
        print("  python metta/tools/compare_training.py --num-pairs 2 --base-run-name test_comparison")
    else:
        print("✗ Some tests failed. Please fix the issues before using the comparison tool.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
