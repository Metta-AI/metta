#!/usr/bin/env python3
"""
Custom linter to check for torch.tensor() calls without explicit dtype parameter.

This script scans Python files and identifies torch.tensor() calls that don't
specify a dtype parameter, which can lead to inconsistent behavior across
different PyTorch versions and platforms.

Usage:
    python devops/lint_tensor_dtype.py [directory]

Examples:
    python devops/lint_tensor_dtype.py metta/
    python devops/lint_tensor_dtype.py tests/
    python devops/lint_tensor_dtype.py .  # scan entire project
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import List, Tuple


class TensorDtypeChecker(ast.NodeVisitor):
    """AST visitor to find torch.tensor() calls without dtype parameter."""

    def __init__(self, filename: str):
        self.filename = filename
        self.issues: List[Tuple[int, str]] = []

    def visit_Call(self, node: ast.Call):
        """Visit function call nodes and check for torch.tensor() calls."""
        # Check if this is a torch.tensor() call
        if self._is_torch_tensor_call(node):
            has_dtype = self._has_dtype_argument(node)
            has_dtype_chained = self._has_chained_dtype(node)

            if not has_dtype and not has_dtype_chained:
                line_num = node.lineno
                self.issues.append((line_num, "torch.tensor() call without explicit dtype parameter"))

        self.generic_visit(node)

    def _is_torch_tensor_call(self, node: ast.Call) -> bool:
        """Check if the call is torch.tensor()."""
        if isinstance(node.func, ast.Attribute):
            # torch.tensor()
            if (
                isinstance(node.func.value, ast.Name) and node.func.value.id == "torch" and node.func.attr == "tensor"
            ):  # Only check for 'tensor', not 'Tensor'
                return True
        return False

    def _has_dtype_argument(self, node: ast.Call) -> bool:
        """Check if the call has a dtype keyword argument."""
        for keyword in node.keywords:
            if keyword.arg == "dtype":
                return True
        return False

    def _has_chained_dtype(self, node: ast.Call) -> bool:
        """Check if there's a chained .to(dtype=...) call.
        Currently, this check is simplified and always returns False as we focus
        on dtype argument directly in torch.tensor().
        A more sophisticated check for chained calls might be needed if rules for
        torch.Tensor() are introduced.
        """
        return False


def check_file(filepath: Path) -> List[Tuple[int, str]]:
    """Check a single Python file for torch.tensor() calls without dtype."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))
        checker = TensorDtypeChecker(str(filepath))
        checker.visit(tree)

        return checker.issues
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)
        return []


def scan_directory(directory: Path, recursive: bool = True) -> int:
    """Scan directory for Python files and check them."""
    issues_found = 0

    if recursive:
        python_files = directory.rglob("*.py")
    else:
        python_files = directory.glob("*.py")

    for filepath in python_files:
        # Skip certain directories
        if any(part in filepath.parts for part in [".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"]):
            continue

        issues = check_file(filepath)
        if issues:
            print(f"\n{filepath}:")
            for line_num, message in issues:
                print(f"  Line {line_num}: {message}")
                issues_found += 1

    return issues_found


def main():
    parser = argparse.ArgumentParser(description="Check for torch.tensor() calls without dtype parameter")
    parser.add_argument("path", nargs="?", default=".", help="Directory or file to check (default: current directory)")
    parser.add_argument("--no-recursive", action="store_true", help="Don't scan subdirectories recursively")

    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        if path.suffix == ".py":
            issues = check_file(path)
            if issues:
                print(f"{path}:")
                for line_num, message in issues:
                    print(f"  Line {line_num}: {message}")
                return len(issues)
        else:
            print(f"Error: {path} is not a Python file")
            return 1
    elif path.is_dir():
        issues_found = scan_directory(path, recursive=not args.no_recursive)

        if issues_found == 0:
            print("✅ No torch.tensor() calls without dtype found!")
        else:
            print(f"\n❌ Found {issues_found} torch.tensor() calls without explicit dtype parameter.")
            print("\nRecommendations:")
            print("- Use dtype=torch.float32 for floating point computations")
            print("- Use dtype=torch.long for integer indices and counters")
            print("- Use dtype=torch.int32 for integer data when memory efficiency is important")

        return min(issues_found, 1)  # Return 1 if any issues found, 0 otherwise
    else:
        print(f"Error: {path} does not exist")
        return 1


if __name__ == "__main__":
    sys.exit(main())
