#!/usr/bin/env python3
# Test runner for C++ stats tracker tests
import os
import subprocess
import sys


def run_test():
    # When run by Bazel, the binary is available in the runfiles
    # The binary is placed in the same directory as this script in the runfiles tree
    test_bin = os.path.join(os.path.dirname(__file__), "test_stats_tracker_bin")

    # Check if the binary exists
    if os.path.exists(test_bin):
        result = subprocess.run([test_bin], capture_output=False)
        return result.returncode

    # If not found, print error message for debugging
    print(f"ERROR: Could not find test binary at: {test_bin}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    print(f"Directory contents: {os.listdir(os.path.dirname(__file__))}")
    return 1


if __name__ == "__main__":
    sys.exit(run_test())
