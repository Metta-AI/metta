# Test runner for C++ grid object tests
import subprocess
import sys
from pathlib import Path


def run_test():
    # Find the test binary in bazel-bin directory
    mettagrid_dir = Path(__file__).parent.parent
    test_bin = mettagrid_dir / "bazel-bin" / "test_grid_object_bin"
    if not test_bin.exists():
        # Try without _bin suffix
        test_bin = mettagrid_dir / "bazel-bin" / "test_grid_object"
    if not test_bin.exists():
        # Try in current directory as fallback
        test_bin = Path("test_grid_object_bin")

    result = subprocess.run([str(test_bin)], capture_output=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_test())
