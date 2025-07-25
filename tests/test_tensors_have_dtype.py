#!/usr/bin/env python3
"""
Unit test to check for torch.tensor() calls without explicit dtype parameter.

This test scans Python files and identifies torch.tensor() calls that don't
specify a dtype parameter, which can lead to inconsistent behavior across
different PyTorch versions and platforms.
"""

import ast
from pathlib import Path
from typing import List, Tuple

from pytest import fail


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


def collect_py_files(root: Path, exclude_dirs: set[str]) -> list[Path]:
    """Collect all Python files from the root directory, excluding specified directories."""
    files = []
    for path in root.rglob("*.py"):
        if any(part in exclude_dirs for part in path.parts):
            continue
        files.append(path)
    return files


def check_file_for_tensor_dtypes(filepath: Path) -> List[Tuple[int, str]]:
    """Check a single Python file for torch.tensor() calls without dtype."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))
        checker = TensorDtypeChecker(str(filepath))
        checker.visit(tree)

        return checker.issues
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that can't be parsed
        return []
    except Exception as e:
        fail(f"Error checking file {filepath}: {e}")


def test_tensors_have_dtype():
    """Test that all torch.tensor() calls in the codebase have explicit dtype parameters."""
    root = Path(__file__).resolve().parents[1]
    exclude_dirs = {
        ".venv",
        "build",
        "build-debug",
        "metta.egg-info",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".uv-cache",
        "node_modules",
        "wandb",  # Exclude wandb logs as they might contain generated code
        "tests",  # Exclude test files as they don't require explicit dtype
    }

    py_files = collect_py_files(root, exclude_dirs)

    # Collect all issues across all files
    all_issues = []
    for filepath in py_files:
        issues = check_file_for_tensor_dtypes(filepath)
        if issues:
            for line_num, message in issues:
                all_issues.append(f"{filepath}:{line_num} - {message}")

    # If any issues are found, fail the test with detailed information
    if all_issues:
        error_message = (
            f"Found {len(all_issues)} torch.tensor() calls without explicit dtype parameter:\n\n"
            + "\n".join(all_issues)
            + "\n\nRecommendations:\n"
            + "- Use dtype=torch.float32 for floating point computations\n"
            + "- Use dtype=torch.long for integer indices and counters\n"
            + "- Use dtype=torch.int32 for integer data when memory efficiency is important"
        )
        raise AssertionError(error_message)


if __name__ == "__main__":
    test_tensors_have_dtype()
