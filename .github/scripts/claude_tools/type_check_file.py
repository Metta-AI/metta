#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mypy>=1.0.0",
#   "pyright>=1.1.300",
# ]
# ///
"""
Type checking tool for Claude reviews.
Provides structured analysis of Python files for missing type annotations.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class TypeAnalyzer(ast.NodeVisitor):
    """Analyze Python AST for missing type annotations."""

    def __init__(self, filename: str):
        self.filename = filename
        self.missing_annotations: List[Dict[str, Any]] = []
        self.current_class: Optional[str] = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function definitions for missing type annotations."""
        issues = []

        # Check if this is a private method
        is_private = node.name.startswith("_")

        # Check if this is a property
        is_property = any(isinstance(dec, ast.Name) and dec.id == "property" for dec in node.decorator_list)

        # Check if this is a very short function (1-3 lines)
        is_short = len(node.body) <= 3

        # Check for missing parameter annotations
        for arg in node.args.args:
            if arg.arg != "self" and arg.annotation is None:
                issues.append({"type": "missing_param_type", "param": arg.arg, "line": arg.lineno, "priority": "high"})

        # Check for missing return type annotation
        if node.returns is None:
            # Determine priority based on function characteristics
            if is_private or is_property or is_short:
                priority = None  # Don't report
            elif self._returns_optional(node):
                priority = "high"  # Functions that might return None
            elif self._is_complex_function(node):
                priority = "medium"  # Complex functions
            else:
                priority = "low"

            if priority:
                issues.append(
                    {
                        "type": "missing_return_type",
                        "line": node.lineno,
                        "priority": priority,
                        "is_private": is_private,
                        "is_property": is_property,
                        "is_short": is_short,
                        "might_return_none": self._returns_optional(node),
                    }
                )

        # Record all issues for this function
        if issues:
            self.missing_annotations.append(
                {"function": node.name, "class": self.current_class, "line": node.lineno, "issues": issues}
            )

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track current class context."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def _returns_optional(self, node: ast.FunctionDef) -> bool:
        """Check if function might return None."""
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return):
                if stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None):
                    return True
        return False

    def _is_complex_function(self, node: ast.FunctionDef) -> bool:
        """Determine if a function is complex enough to warrant return type annotation."""
        # Multiple return statements
        returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
        if len(returns) > 1:
            return True

        # Function longer than 10 lines
        if len(node.body) > 10:
            return True

        # Contains nested functions or classes
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.ClassDef)):
                return True

        return False


def analyze_file(filepath: Path) -> Dict[str, Any]:
    """Analyze a Python file for missing type annotations."""
    try:
        with open(filepath, "r") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))
        analyzer = TypeAnalyzer(str(filepath))
        analyzer.visit(tree)

        return {"file": str(filepath), "missing_annotations": analyzer.missing_annotations, "error": None}
    except Exception as e:
        return {"file": str(filepath), "missing_annotations": [], "error": str(e)}


def run_mypy(files: List[str]) -> Dict[str, Any]:
    """Run mypy on the specified files."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--no-error-summary", "--show-column-numbers"] + files,
            capture_output=True,
            text=True,
        )

        return {"tool": "mypy", "output": result.stdout, "errors": result.stderr, "return_code": result.returncode}
    except Exception as e:
        return {"tool": "mypy", "output": "", "errors": str(e), "return_code": -1}


def run_pyright(files: List[str]) -> Dict[str, Any]:
    """Run pyright on the specified files."""
    try:
        result = subprocess.run(["pyright", "--outputjson"] + files, capture_output=True, text=True)

        return {"tool": "pyright", "output": result.stdout, "errors": result.stderr, "return_code": result.returncode}
    except Exception as e:
        return {"tool": "pyright", "output": "", "errors": str(e), "return_code": -1}


def filter_high_value_annotations(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter to only include high-value missing annotations."""
    filtered = []

    for func_info in annotations:
        high_value_issues = []

        for issue in func_info["issues"]:
            # Always include missing parameter types
            if issue["type"] == "missing_param_type":
                high_value_issues.append(issue)
            # Include return types based on priority
            elif issue["type"] == "missing_return_type" and issue["priority"] in ["high", "medium"]:
                high_value_issues.append(issue)

        if high_value_issues:
            filtered.append({**func_info, "issues": high_value_issues})

    return filtered


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: type_check.py <command> [args...]")
        print("Commands:")
        print("  analyze <file.py>     - Analyze file for missing annotations")
        print("  mypy <file.py>        - Run mypy on file")
        print("  pyright <file.py>     - Run pyright on file")
        print("  check <file.py>       - Run all checks and summarize")
        sys.exit(1)

    command = sys.argv[1]

    if command == "analyze" and len(sys.argv) >= 3:
        # Analyze Python files for missing annotations
        results = []
        for filepath in sys.argv[2:]:
            result = analyze_file(Path(filepath))
            # Filter to high-value annotations only
            result["missing_annotations"] = filter_high_value_annotations(result["missing_annotations"])
            results.append(result)

        print(json.dumps(results, indent=2))

    elif command == "mypy" and len(sys.argv) >= 3:
        # Run mypy
        result = run_mypy(sys.argv[2:])
        print(json.dumps(result, indent=2))

    elif command == "pyright" and len(sys.argv) >= 3:
        # Run pyright
        result = run_pyright(sys.argv[2:])
        print(json.dumps(result, indent=2))

    elif command == "check" and len(sys.argv) >= 3:
        # Comprehensive check
        files = sys.argv[2:]

        # Analyze with AST
        analysis_results = []
        for filepath in files:
            result = analyze_file(Path(filepath))
            result["missing_annotations"] = filter_high_value_annotations(result["missing_annotations"])
            analysis_results.append(result)

        # Run type checkers
        mypy_result = run_mypy(files)
        pyright_result = run_pyright(files)

        # Combine results
        summary = {
            "files_analyzed": len(files),
            "total_missing_annotations": sum(len(r["missing_annotations"]) for r in analysis_results),
            "analysis": analysis_results,
            "mypy": mypy_result,
            "pyright": pyright_result,
        }

        print(json.dumps(summary, indent=2))

    else:
        print(f"Unknown command or missing arguments: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
