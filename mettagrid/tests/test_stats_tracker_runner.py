# Test runner for C++ stats tracker tests
import subprocess
import sys
import os


def run_test():
    # Find the test binary in bazel runfiles
    test_bin = os.path.join(os.path.dirname(__file__), "test_stats_tracker_bin")
    if not os.path.exists(test_bin):
        # Try bazel runfiles location
        test_bin = "test_stats_tracker_bin"

    result = subprocess.run([test_bin], capture_output=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_test())