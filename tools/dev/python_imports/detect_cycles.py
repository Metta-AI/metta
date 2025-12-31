#!/usr/bin/env python3
"""Detect circular import dependencies in Python codebase.

Usage:
    python detect_cycles.py [--path PATH] [--output OUTPUT.json]

Analyzes Python files to build an import dependency graph and identify
circular dependencies using strongly connected components.

If output path is not specified, writes report to
`tools/dev/python_imports/cycles_report.json`.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImportInfo:
    imported_module: str
    is_type_checking: bool
    line_number: int
    import_type: str  # 'symbol' or 'module'
    is_local_import: bool  # True if inside function/method, False if module-level


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to extract import information."""

    def __init__(self, module_path: str):
        self.module_path = module_path
        self.imports: list[ImportInfo] = []
        self.in_type_checking = False
        self.type_checking_level = 0
        self.function_depth = 0  # Track nesting level (0 = module level)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track function/method depth."""
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track async function/method depth."""
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        """Track if we're inside a TYPE_CHECKING block."""
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            self.in_type_checking = True
            self.type_checking_level += 1
            self.generic_visit(node)
            self.type_checking_level -= 1
            if self.type_checking_level == 0:
                self.in_type_checking = False
        else:
            self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements: import X as Y"""
        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    imported_module=alias.name,
                    is_type_checking=self.in_type_checking,
                    line_number=node.lineno,
                    import_type="module",
                    is_local_import=self.function_depth > 0,
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from imports: from X import Y

        Handles both absolute and relative imports.
        """
        # Handle relative imports (from . import X or from ..module import Y)
        if node.level > 0:
            module_parts = self.module_path.split(".")

            if node.level <= len(module_parts):
                # Valid relative import within package
                base_path = ".".join(module_parts[: -node.level])
                if node.module:
                    # from .module import X - imports from a specific submodule
                    imported = f"{base_path}.{node.module}" if base_path else node.module
                    self.imports.append(
                        ImportInfo(
                            imported_module=imported,
                            is_type_checking=self.in_type_checking,
                            line_number=node.lineno,
                            import_type="symbol",
                            is_local_import=self.function_depth > 0,
                        )
                    )
                else:
                    # from . import X, Y - imports sibling modules
                    # Each name is a separate module dependency
                    for alias in node.names:
                        if alias.name == "*":
                            # from . import * - depends on parent package
                            imported = base_path if base_path else self.module_path
                        else:
                            # from . import foo -> depends on pkg.foo
                            imported = f"{base_path}.{alias.name}" if base_path else alias.name
                        self.imports.append(
                            ImportInfo(
                                imported_module=imported,
                                is_type_checking=self.in_type_checking,
                                line_number=node.lineno,
                                import_type="symbol",
                                is_local_import=self.function_depth > 0,
                            )
                        )
            else:
                # Invalid: too many levels up, skip this import
                # (This would cause a Python error at runtime)
                self.generic_visit(node)
                return
        elif node.module:
            # Absolute import: from X import Y
            imported = node.module
            self.imports.append(
                ImportInfo(
                    imported_module=imported,
                    is_type_checking=self.in_type_checking,
                    line_number=node.lineno,
                    import_type="symbol",
                    is_local_import=self.function_depth > 0,
                )
            )
        else:
            # No module specified (shouldn't happen), skip
            self.generic_visit(node)
            return

        self.generic_visit(node)


def parse_file(file_path: Path, package_root: Path) -> tuple[str, list[ImportInfo]]:
    """Parse a Python file and extract import details.

    Returns:
        (module_path, imports) tuple
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))

        relative = file_path.relative_to(package_root)
        module_parts = list(relative.parts[:-1])  # Remove filename
        if relative.stem != "__init__":
            module_parts.append(relative.stem)
        module_path = ".".join(module_parts) if module_parts else "__main__"

        analyzer = ImportAnalyzer(module_path)
        analyzer.visit(tree)

        return module_path, analyzer.imports
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return "", []


def build_dependency_graph(root_path: Path) -> dict[str, dict]:
    """Build import dependency graph for Python files.

    Returns:
        Dict mapping module paths to their import information
    """
    graph = defaultdict(lambda: {"imports": [], "file": None})

    python_files = []
    for pattern in ["**/*.py"]:
        python_files.extend(root_path.rglob(pattern))

    # Exclude common non-package directories
    excluded_dirs = {
        ".venv",
        "venv",
        "node_modules",
        ".git",
        "__pycache__",
        ".pytest_cache",
        "build",
        "dist",
        ".eggs",
        ".bazel_output",
        "bazel-out",
        "external",
    }

    python_files = [f for f in python_files if not any(excluded in f.parts for excluded in excluded_dirs)]

    print(f"Analyzing {len(python_files)} Python files...")

    for file_path in python_files:
        module_path, imports = parse_file(file_path, root_path)
        if module_path:
            graph[module_path]["imports"] = imports
            graph[module_path]["file"] = str(file_path.relative_to(root_path))

    return dict(graph)


def find_cycles(graph: dict[str, dict]) -> list[list[str]]:
    """Find all cycles in the import graph using Tarjan's algorithm.

    Only considers module-level runtime imports (excludes TYPE_CHECKING and local imports)
    as those don't create circular dependency issues.

    Args:
        graph: Import dependency graph mapping modules to their import info

    Returns:
        List of cycles, where each cycle is a list of module paths forming a cycle
    """
    # Build adjacency list from import edges
    # Only include module-level runtime imports (not TYPE_CHECKING or local)
    adjacency = defaultdict(set)

    for module, data in graph.items():
        for imp in data["imports"]:
            # Skip imports that don't create runtime cycles
            if imp.is_type_checking or imp.is_local_import:
                continue

            imported = imp.imported_module

            # Add edge if the exact imported module exists in our graph.
            if imported in graph:
                adjacency[module].add(imported)

    # Tarjan's algorithm for finding strongly connected components (SCCs)
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = defaultdict(bool)
    sccs = []

    def strongconnect(node: str) -> None:
        """Recursive DFS for Tarjan's algorithm."""
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True

        # Consider successors
        for neighbor in adjacency.get(node, []):
            if neighbor not in index:
                # Neighbor not yet visited; recurse
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif on_stack[neighbor]:
                # Neighbor is in the current SCC
                lowlinks[node] = min(lowlinks[node], index[neighbor])

        # If node is a root node, form an SCC
        if lowlinks[node] == index[node]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == node:
                    break
            # Only report SCCs with more than one node (actual cycles)
            # Single-node SCCs are not circular dependencies
            if len(scc) > 1:
                sccs.append(scc)

    for node in graph.keys():
        if node not in index:
            strongconnect(node)

    return sccs


def classify_cycles(cycles: list[list[str]], graph: dict[str, dict]) -> dict:
    """Classify cycles by type and severity.

    Adds metadata to help human reviewers identify false positives:
    - Import edges showing direction of dependencies
    - Whether imports are symbol imports (inheritance) vs module imports
    """

    def get_package(module: str) -> str:
        """Get top-level package name."""
        return module.split(".")[0]

    cross_package = []
    same_package = []

    for cycle in cycles:
        packages = {get_package(m) for m in cycle}

        # Extract import edges within this cycle to show dependency direction
        edges = []
        for module in cycle:
            if module not in graph:
                continue
            for imp in graph[module]["imports"]:
                # Only include runtime imports (TYPE_CHECKING already excluded)
                if imp.is_type_checking or imp.is_local_import:
                    continue
                # Check if this import is to another module in the cycle
                if imp.imported_module in cycle or any(m.startswith(imp.imported_module + ".") for m in cycle):
                    edges.append(
                        {
                            "from": module,
                            "to": imp.imported_module,
                            "type": imp.import_type,  # 'symbol' or 'module'
                            "line": imp.line_number,
                        }
                    )

        cycle_info = {
            "modules": cycle,
            "packages": sorted(packages),
            "files": [graph[m]["file"] for m in cycle if m in graph],
            "import_edges": edges,
            "potential_false_positive": len(edges) == 2
            and edges[0]["type"] == "symbol"
            and edges[1]["type"] == "symbol",
            "review_hint": "Review consideration: Check if one direction is from base to derived (inheritance only)"
            if len(edges) == 2
            else None,
        }

        if len(packages) > 1:
            cycle_info["severity"] = "high"
            cross_package.append(cycle_info)
        else:
            cycle_info["severity"] = "medium"
            same_package.append(cycle_info)

    return {
        "cross_package_cycles": cross_package,
        "same_package_cycles": same_package,
        "total_cycles": len(cycles),
    }


def analyze_type_checking_usage(graph: dict[str, dict]) -> dict:
    """Analyze TYPE_CHECKING block usage."""
    files_with_type_checking = []

    for module, data in graph.items():
        type_checking_imports = [imp for imp in data["imports"] if imp.is_type_checking]
        if type_checking_imports:
            files_with_type_checking.append(
                {
                    "module": module,
                    "file": data["file"],
                    "count": len(type_checking_imports),
                    "imports": [
                        {
                            "module": imp.imported_module,
                            "line": imp.line_number,
                            "type": imp.import_type,
                        }
                        for imp in type_checking_imports
                    ],
                }
            )

    return {
        "files_with_type_checking": files_with_type_checking,
        "total_files": len(files_with_type_checking),
    }


def analyze_local_imports(graph: dict[str, dict]) -> dict:
    """Analyze local (function-level) import usage.

    These violate our style guide: all imports should be at module level.
    """
    files_with_local_imports = []

    for module, data in graph.items():
        local_imports = [imp for imp in data["imports"] if imp.is_local_import]
        if local_imports:
            files_with_local_imports.append(
                {
                    "module": module,
                    "file": data["file"],
                    "count": len(local_imports),
                    "imports": [
                        {
                            "module": imp.imported_module,
                            "line": imp.line_number,
                            "type": imp.import_type,
                            "in_type_checking": imp.is_type_checking,
                        }
                        for imp in local_imports
                    ],
                }
            )

    return {
        "files_with_local_imports": files_with_local_imports,
        "total_files": len(files_with_local_imports),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect circular import dependencies")
    parser.add_argument(
        "--path", type=Path, default=Path.cwd(), help="Root path to analyze (default: current directory)"
    )
    parser.add_argument("--output", type=Path, help="Output JSON file (default: print to stdout)")
    parser.add_argument("--focus", type=str, help="Focus on specific packages (comma-separated)")

    args = parser.parse_args()

    output_path = args.output or Path.cwd() / "tools/dev/python_imports/cycles_report.json"

    if not args.path.exists():
        print(f"Error: Path {args.path} does not exist", file=sys.stderr)
        return 1

    graph = build_dependency_graph(args.path)

    cycles = find_cycles(graph)

    cycle_info = classify_cycles(cycles, graph)

    type_checking_info = analyze_type_checking_usage(graph)

    local_imports_info = analyze_local_imports(graph)

    report = {
        "summary": {
            "total_modules": len(graph),
            "total_cycles": cycle_info["total_cycles"],
            "cross_package_cycles": len(cycle_info["cross_package_cycles"]),
            "same_package_cycles": len(cycle_info["same_package_cycles"]),
            "files_with_type_checking": type_checking_info["total_files"],
            "files_with_local_imports": local_imports_info["total_files"],
        },
        "cycles": cycle_info,
        "type_checking": type_checking_info,
        "local_imports": local_imports_info,
    }

    # Filter by focus packages if specified
    if args.focus:
        focus_packages = set(pkg.strip() for pkg in args.focus.split(","))
        report["focus_packages"] = list(focus_packages)

    output_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to {output_path}")
    print("\nSummary:")
    print(f"  Total modules analyzed: {report['summary']['total_modules']}")
    print(f"  Circular dependencies found: {report['summary']['total_cycles']}")
    print(f"    Cross-package: {report['summary']['cross_package_cycles']}")
    print(f"    Same-package: {report['summary']['same_package_cycles']}")
    print(f"  Files with TYPE_CHECKING: {report['summary']['files_with_type_checking']}")
    print(f"  Files with local imports: {report['summary']['files_with_local_imports']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
