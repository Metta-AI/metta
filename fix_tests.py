#!/usr/bin/env python3
"""
Quick script to fix common test issues in the metta test suite.
"""

import os
import re


def fix_test_file(filepath):
    """Fix common test issues in a test file."""
    print(f"Fixing {filepath}...")

    with open(filepath, "r") as f:
        content = f.read()

    original_content = content

    # Fix LinearModule instantiation patterns
    # Pattern: LinearModule(int, int, "key", "key") -> LinearModule(in_features=int, out_features=int, in_key="key", out_key="key")
    content = re.sub(
        r'LinearModule\((\d+),\s*(\d+),\s*"([^"]+)",\s*"([^"]+)"\)',
        r'LinearModule(in_features=\1, out_features=\2, in_key="\3", out_key="\4")',
        content,
    )

    # Fix LinearModule instantiation without keys
    # Pattern: LinearModule(int, int) -> LinearModule(in_features=int, out_features=int, in_key="input", out_key="output")
    content = re.sub(
        r"LinearModule\((\d+),\s*(\d+)\)(?!\s*,)",
        r'LinearModule(in_features=\1, out_features=\2, in_key="input", out_key="output")',
        content,
    )

    # Fix key naming: replace "x" with "input" in TensorDict creation
    content = re.sub(r'TensorDict\(\{"x":\s*([^}]+)\}', r'TensorDict({"input": \1}', content)

    # Fix forward calls that expect two arguments
    content = re.sub(r'container\.forward\("([^"]+)",\s*([^)]+)\)', r'container.execute_component("\1", \2)', content)

    # Fix missing input_shapes parameter in LSTM tests
    content = re.sub(r"LSTMModule\((\d+),\s*(\d+)", r"LSTMModule(input_size=\1, hidden_size=\2", content)

    if content != original_content:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"  -> Fixed {filepath}")
        return True
    else:
        print(f"  -> No changes needed for {filepath}")
        return False


def main():
    """Fix all test files."""
    test_dir = "tests/agent/lib"

    # Files that likely need fixing based on the test failures
    test_files = [
        "test_performance.py",
        "test_integration.py",
        "test_serialization.py",
        "test_edge_cases.py",
        "test_lstm_module.py",
        "test_metta_modules.py",
    ]

    fixed_count = 0
    for test_file in test_files:
        filepath = os.path.join(test_dir, test_file)
        if os.path.exists(filepath):
            if fix_test_file(filepath):
                fixed_count += 1
        else:
            print(f"File not found: {filepath}")

    print(f"\nFixed {fixed_count} files.")


if __name__ == "__main__":
    main()
