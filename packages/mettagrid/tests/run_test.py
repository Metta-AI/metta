#!/usr/bin/env python3
# Generic test runner for C++ binaries in Bazel runfiles
import argparse
import os
import subprocess
import sys
from typing import Optional


def find_test_binary(bin_name: str) -> Optional[str]:
    """Locate the test binary in the Bazel runfiles alongside this script."""
    script_dir = os.path.dirname(__file__)
    candidate = os.path.join(script_dir, bin_name)
    if os.path.exists(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a C++ test binary from Bazel runfiles")
    parser.add_argument(
        "--bin", dest="bin_name", default=os.environ.get("TEST_BIN"), help="Name of the test binary to execute"
    )
    args = parser.parse_args()

    if not args.bin_name:
        print("ERROR: No test binary specified. Pass --bin=<binary_name> or set TEST_BIN env var.")
        return 2

    test_bin = find_test_binary(args.bin_name)
    if not test_bin:
        print(f"ERROR: Could not find test binary: {args.bin_name}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Script location: {__file__}")
        try:
            contents = os.listdir(os.path.dirname(__file__))
        except Exception as e:
            contents = [f"<error listing dir: {e}>"]
        print(f"Directory contents: {contents}")
        return 1

    result = subprocess.run([test_bin], capture_output=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
