#!/usr/bin/env python3
"""Analyze Python codebase architecture for import refactoring.

Usage:
    python analyze_architecture.py [--path PATH] [--output OUTPUT.json]

Identifies types that should be extracted to types.py, __init__.py files that
should be simplified, and provides recommendations for import refactoring.

If output path is not specified, writes to
`tools/dev/python_imports/architecture_report.json`.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import detect_cycles


@dataclass
class TypeDefinition:
    name: str
    module: str
    file: str
    line_number: int
    type_kind: str  # 'class', 'TypeAlias', 'TypedDict', 'dataclass', 'Protocol'
    is_exported: bool
    usage_count: int = 0
    used_in_modules: list[str] | None = None


@dataclass
class InitAnalysis:
    module: str
    file: str
    total_lines: int
    import_lines: int
    has_getattr: bool
    has_lazy_loading: bool
    exports: list[str]
    complexity_score: float
    recommendation: str  # 'keep', 'simplify', 'empty'
    reason: str


class TypeExtractor(ast.NodeVisitor):
    """AST visitor to extract type definitions from Python files."""

    def __init__(self, module_path: str):
        self.module_path = module_path
        self.types: list[TypeDefinition] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        # Get decorators on this class
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                decorators.append(decorator.func.id)

        # Determine type kind
        type_kind = "class"
        if any("Protocol" in base.id if isinstance(base, ast.Name) else False for base in node.bases):
            type_kind = "Protocol"
        elif any("TypedDict" in base.id if isinstance(base, ast.Name) else False for base in node.bases):
            type_kind = "TypedDict"
        elif "dataclass" in decorators:
            type_kind = "dataclass"

        self.types.append(
            TypeDefinition(
                name=node.name,
                module=self.module_path,
                file="",  # Will be set by caller
                line_number=node.lineno,
                type_kind=type_kind,
                is_exported=not node.name.startswith("_"),
                used_in_modules=[],
            )
        )

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit annotated assignments (type aliases)."""
        if isinstance(node.target, ast.Name):
            # Check if this is a type alias
            if isinstance(node.annotation, ast.Subscript):
                if isinstance(node.annotation.value, ast.Name):
                    if node.annotation.value.id in ("TypeAlias", "Type"):
                        self.types.append(
                            TypeDefinition(
                                name=node.target.id,
                                module=self.module_path,
                                file="",
                                line_number=node.lineno,
                                type_kind="TypeAlias",
                                is_exported=not node.target.id.startswith("_"),
                                used_in_modules=[],
                            )
                        )
        self.generic_visit(node)


def analyze_import_weight(file_path: Path) -> str:
    """Determine if a module is expensive to import based on known import patterns.

    Binary classification:
    - 'heavy': Justifies lazy-loading (torch, gymnasium, pufferlib, pandas, scipy, boto3, etc.)
    - 'light': Safe to import directly (everything else)

    To add new heavy patterns, update the heavy_patterns list below.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "light"

    # Heavy patterns: single occurrence justifies lazy-loading
    heavy_patterns = [
        r"\bimport torch\b",
        r"\bfrom torch\b",
        r"\bimport tensorflow\b",
        r"\bfrom tensorflow\b",
        r"\bimport jax\b",
        r"\bfrom jax\b",
        r"\bimport pufferlib\b",
        r"\bfrom pufferlib\b",
        r"\bimport gymnasium\b",
        r"\bfrom gymnasium\b",
        r"\bimport pandas\b",
        r"\bfrom pandas\b",
        r"\bimport scipy\b",
        r"\bfrom scipy\b",
        r"\bimport sqlalchemy\b",
        r"\bfrom sqlalchemy\b",
        r"\bimport boto3\b",
        r"\bfrom boto3\b",
        r"\bimport botocore\b",
        r"\bfrom botocore\b",
        r"\bimport transformers\b",
        r"\bfrom transformers\b",
        r"\bclass\s+\w+\s*\(.*nn\.Module",  # Neural network components
    ]

    # Any heavy import justifies lazy-loading
    if any(re.search(pattern, content) for pattern in heavy_patterns):
        return "heavy"

    return "light"


class InitFileAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.imports: list[str] = []
        self.exports: list[str] = []
        self.has_getattr = False
        self.has_lazy_loading = False
        self.import_line_count = 0

    def visit_Import(self, node: ast.Import) -> None:
        self.import_line_count += 1
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.import_line_count += 1
        if node.module:
            for alias in node.names:
                self.imports.append(f"{node.module}.{alias.name}")
                if alias.name != "*":
                    self.exports.append(alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Detect __getattr__ for lazy loading."""
        if node.name == "__getattr__":
            self.has_getattr = True
            self.has_lazy_loading = True
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        """Detect optional dependencies."""
        # Check if this is an import try/except
        for child in node.body:
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                self.has_lazy_loading = True
                break
        self.generic_visit(node)


def extract_types_from_file(file_path: Path, package_root: Path) -> list[TypeDefinition]:
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))

        # Convert file path to module path
        relative = file_path.relative_to(package_root)
        module_parts = list(relative.parts[:-1])
        if relative.stem != "__init__":
            module_parts.append(relative.stem)
        module_path = ".".join(module_parts) if module_parts else "__main__"

        extractor = TypeExtractor(module_path)
        extractor.visit(tree)

        for type_def in extractor.types:
            type_def.file = str(relative)

        return extractor.types
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return []


def analyze_init_file(file_path: Path, package_root: Path) -> InitAnalysis:
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))

        relative = file_path.relative_to(package_root)
        module_parts = list(relative.parts[:-1])
        module_path = ".".join(module_parts) if module_parts else "__main__"

        analyzer = InitFileAnalyzer()
        analyzer.visit(tree)

        total_lines = len(content.splitlines())

        complexity = analyzer.import_line_count / max(total_lines, 1)

        package_dir = file_path.parent
        submodule_weights: dict[str, str] = {}
        for py_file in package_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            weight = analyze_import_weight(py_file)
            submodule_weights[py_file.stem] = weight

        heavy_modules = [m for m, w in submodule_weights.items() if w == "heavy"]

        # Detect if this is a public package or internal module
        # Public packages: packages/mettagrid, packages/cortex, etc.
        # Internal modules: metta/rl, metta/sim, etc.
        is_public_package = "packages" in relative.parts

        # Determine recommendation based on package type and lazy loading status
        if is_public_package:
            # PUBLIC PACKAGE: Encourage thoughtful exports with lazy loading
            # Check for empty/minimal files first
            if len(analyzer.exports) == 0 and total_lines < 10 and not analyzer.has_lazy_loading:
                recommendation = "none"
                reason = "Already minimal or empty"
            elif analyzer.has_lazy_loading or analyzer.has_getattr:
                # Has lazy loading - is it justified?
                if len(submodule_weights) == 0:
                    recommendation = "keep"
                    reason = "Public package with lazy loading (no submodules to analyze)"
                elif len(heavy_modules) == 0:
                    # No heavy modules - lazy loading not justified
                    recommendation = "simplify"
                    reason = "Lazy loading not justified: no heavy submodules"
                else:
                    # Heavy modules present - lazy loading is justified
                    recommendation = "keep"
                    reason = (
                        "Public package with justified lazy loading: "
                        f"{len(heavy_modules)} heavy modules ({heavy_modules})"
                    )
            elif len(heavy_modules) > 0:
                # No lazy loading but has heavy modules - should add lazy loading
                recommendation = "add_lazy_loading"
                reason = f"Public package with heavy submodules ({heavy_modules}) should use lazy loading"
            elif complexity > 0.8:  # More than 80% imports
                recommendation = "simplify"
                reason = f"Public package with high import complexity ({complexity:.0%})"
            elif len(analyzer.exports) > 20:
                recommendation = "simplify"
                reason = f"Public package exports too many symbols ({len(analyzer.exports)})"
            else:
                recommendation = "keep"
                reason = "Public package with reasonable complexity"
        else:
            # INTERNAL MODULE: Prefer empty/minimal __init__.py
            # Check for empty/minimal files first
            if len(analyzer.exports) == 0 and total_lines < 10:
                recommendation = "empty"
                reason = "Already minimal (recommended for internal modules)"
            elif analyzer.has_lazy_loading or analyzer.has_getattr:
                # Has lazy loading - generally not needed for internal modules
                if len(heavy_modules) > 0:
                    # Exception: heavy modules with lazy loading is acceptable
                    recommendation = "keep"
                    reason = (
                        "Internal module with justified lazy loading: "
                        f"{len(heavy_modules)} heavy modules ({heavy_modules})"
                    )
                else:
                    recommendation = "simplify"
                    reason = "Internal modules should be empty unless justified (has lazy loading but no heavy modules)"
            elif len(analyzer.exports) > 0:
                # Has exports - recommend simplification
                recommendation = "simplify"
                reason = (
                    f"Internal module with {len(analyzer.exports)} exports - "
                    "prefer empty __init__.py (users can import from specific modules)"
                )
            else:
                recommendation = "empty"
                reason = "Internal module - empty __init__.py is recommended"

        return InitAnalysis(
            module=module_path,
            file=str(relative),
            total_lines=total_lines,
            import_lines=analyzer.import_line_count,
            has_getattr=analyzer.has_getattr,
            has_lazy_loading=analyzer.has_lazy_loading,
            exports=analyzer.exports,
            complexity_score=complexity,
            recommendation=recommendation,
            reason=reason,
        )
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return InitAnalysis(
            module="",
            file=str(file_path),
            total_lines=0,
            import_lines=0,
            has_getattr=False,
            has_lazy_loading=False,
            exports=[],
            complexity_score=0.0,
            recommendation="keep",
            reason=f"Parse error: {e}",
        )


def analyze_type_usage(types: list[TypeDefinition], root_path: Path) -> None:
    """Analyze how types are used across the codebase.

    Optimized to compile regex patterns once and use simple string containment
    as a fast pre-filter before running regex.
    """
    # Find all Python files
    python_files = []
    for pattern in ["**/*.py"]:
        python_files.extend(root_path.rglob(pattern))

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

    type_patterns = {}
    for type_def in types:
        type_patterns[type_def.name] = re.compile(rf"\b{re.escape(type_def.name)}\b")

    for file_path in python_files:
        try:
            content = file_path.read_text(encoding="utf-8")

            relative = file_path.relative_to(root_path)
            module_parts = list(relative.parts[:-1])
            if relative.stem != "__init__":
                module_parts.append(relative.stem)
            module_path = ".".join(module_parts) if module_parts else "__main__"

            for type_def in types:
                if type_def.module == module_path:
                    continue  # Skip same module

                if type_def.name not in content:
                    continue

                # Slower but accurate: word boundary regex
                if type_patterns[type_def.name].search(content):
                    type_def.usage_count += 1
                    if type_def.used_in_modules is not None:
                        type_def.used_in_modules.append(module_path)

        except (UnicodeDecodeError, OSError):
            continue


def recommend_type_extraction(types: list[TypeDefinition], root_path: Path, cycle_info: dict) -> dict:
    """Recommend which types should be extracted to shared type files.

    Groups types by second-level package (e.g., 'metta.rl', 'metta.agent')
    and checks for existing shared locations (types.py, common.py, common/).
    Cross-references with cycle detection to highlight extractions that break cycles.

    Args:
        types: List of type definitions found in codebase
        root_path: Root path of the project
        cycle_info: Cycle detection results from detect_cycles.py

    Returns:
        Dict mapping packages to type extraction recommendations
    """

    # Extract modules involved in cycles and their details
    modules_in_cycles = set()
    module_cycle_info = {}  # module -> list of cycles it's in

    all_cycles = cycle_info.get("cross_package_cycles", []) + cycle_info.get("same_package_cycles", [])
    for cycle in all_cycles:
        for module in cycle["modules"]:
            modules_in_cycles.add(module)
            if module not in module_cycle_info:
                module_cycle_info[module] = []
            module_cycle_info[module].append(
                {
                    "modules": cycle["modules"],
                    "potential_false_positive": cycle.get("potential_false_positive", False),
                    "review_hint": cycle.get("review_hint"),
                }
            )

    packages = defaultdict(list)
    for type_def in types:
        if type_def.module:
            parts = type_def.module.split(".")
            package = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
            packages[package].append(type_def)

    recommendations = {}

    for package, package_types in packages.items():
        cross_module_types = [t for t in package_types if t.usage_count >= 2 and t.is_exported]

        if not cross_module_types:
            continue

        # Check for existing shared type locations
        package_dir = root_path / package.replace(".", "/")

        existing_locations = []
        if (package_dir / "types.py").exists():
            existing_locations.append("types.py")
        if (package_dir / "common.py").exists():
            existing_locations.append("common.py")
        if (package_dir / "common" / "__init__.py").exists():
            existing_locations.append("common/__init__.py")
        if (package_dir / "common" / "types.py").exists():
            existing_locations.append("common/types.py")

        # Determine recommendation based on existing files
        # Prefer types.py > common/types.py > common.py > common/__init__.py
        if existing_locations:
            if "types.py" in existing_locations:
                target_file = "types.py"
                action = "use_existing"
            elif "common/types.py" in existing_locations:
                target_file = "common/types.py"
                action = "use_existing"
            elif "common.py" in existing_locations:
                target_file = "common.py"
                action = "use_existing"
            else:  # common/__init__.py
                target_file = "common/__init__.py"
                action = "use_existing"

            reason = f"Use existing {target_file} for shared types"
            if len(existing_locations) > 1:
                reason += f" (consolidate from: {', '.join(existing_locations)})"
        else:
            # No existing shared location - recommend creating types.py
            target_file = "types.py"
            action = "create_new"
            reason = "Create new types.py for shared types"

        # Check if any types are in modules involved in cycles
        types_in_cycles = [t for t in cross_module_types if t.module in modules_in_cycles]

        recommendations[package] = {
            "target_file": f"{package.replace('.', '/')}/{target_file}",
            "action": action,
            "reason": reason,
            "existing_locations": existing_locations,
            "has_types_in_cycles": len(types_in_cycles) > 0,
            "types_to_extract": [
                {
                    "name": t.name,
                    "current_location": t.file,
                    "usage_count": t.usage_count,
                    "type_kind": t.type_kind,
                    "unique_modules_count": len(set(t.used_in_modules)) if t.used_in_modules else 0,
                    "in_cycle": t.module in modules_in_cycles,
                    "cycle_details": module_cycle_info.get(t.module) if t.module in modules_in_cycles else None,
                }
                for t in sorted(cross_module_types, key=lambda x: x.usage_count, reverse=True)
            ],
        }

    return recommendations


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze codebase architecture for import refactoring")
    parser.add_argument(
        "--path", type=Path, default=Path.cwd(), help="Root path to analyze (default: current directory)"
    )
    parser.add_argument("--output", type=Path, help="Output JSON file (default: print to stdout)")

    args = parser.parse_args()

    output_path = args.output or Path.cwd() / "tools/dev/python_imports/architecture_report.json"

    if not args.path.exists():
        print(f"Error: Path {args.path} does not exist", file=sys.stderr)
        return 1

    print("Analyzing Python files...", file=sys.stderr)

    python_files = []
    for pattern in ["**/*.py"]:
        python_files.extend(args.path.rglob(pattern))

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

    print(f"Extracting types from {len(python_files)} files...", file=sys.stderr)
    all_types = []
    for file_path in python_files:
        types = extract_types_from_file(file_path, args.path)
        all_types.extend(types)

    print("Analyzing type usage...", file=sys.stderr)
    analyze_type_usage(all_types, args.path)

    # Detect circular dependencies to cross-reference with type extraction
    print("Detecting circular dependencies...", file=sys.stderr)
    graph = detect_cycles.build_dependency_graph(args.path)
    cycles = detect_cycles.find_cycles(graph)
    cycle_info = detect_cycles.classify_cycles(cycles, graph)

    print("Analyzing __init__.py files...", file=sys.stderr)
    init_files = [f for f in python_files if f.name == "__init__.py"]
    init_analyses = []
    for init_file in init_files:
        analysis = analyze_init_file(init_file, args.path)
        if analysis.module:
            init_analyses.append(analysis)

    print("Generating recommendations...", file=sys.stderr)
    type_recommendations = recommend_type_extraction(all_types, args.path, cycle_info)

    report = {
        "summary": {
            "total_files": len(python_files),
            "total_types": len(all_types),
            "init_files_analyzed": len(init_analyses),
            "init_files_to_simplify": len([a for a in init_analyses if a.recommendation == "simplify"]),
            "init_files_to_empty": len([a for a in init_analyses if a.recommendation == "empty"]),
            "init_files_needing_lazy_loading": len(
                [a for a in init_analyses if a.recommendation == "add_lazy_loading"]
            ),
            "packages_needing_types_file": len(type_recommendations),
        },
        "type_extraction_recommendations": type_recommendations,
        "init_file_analysis": [
            {
                "module": a.module,
                "file": a.file,
                "total_lines": a.total_lines,
                "import_lines": a.import_lines,
                "exports": a.exports,
                "has_lazy_loading": a.has_lazy_loading,
                "complexity_score": round(a.complexity_score, 2),
                "recommendation": a.recommendation,
                "reason": a.reason,
            }
            for a in sorted(init_analyses, key=lambda x: x.complexity_score, reverse=True)
            if a.recommendation != "none"
        ],
        "type_definitions_by_kind": {
            "class": len([t for t in all_types if t.type_kind == "class"]),
            "dataclass": len([t for t in all_types if t.type_kind == "dataclass"]),
            "Protocol": len([t for t in all_types if t.type_kind == "Protocol"]),
            "TypedDict": len([t for t in all_types if t.type_kind == "TypedDict"]),
            "TypeAlias": len([t for t in all_types if t.type_kind == "TypeAlias"]),
        },
    }

    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {output_path}", file=sys.stderr)
    print("\nSummary:", file=sys.stderr)
    print(f"  Files analyzed: {report['summary']['total_files']}", file=sys.stderr)
    print(f"  Types found: {report['summary']['total_types']}", file=sys.stderr)
    print(f"  __init__.py files to simplify: {report['summary']['init_files_to_simplify']}", file=sys.stderr)
    print(
        f"  __init__.py needing lazy loading: {report['summary']['init_files_needing_lazy_loading']}",
        file=sys.stderr,
    )
    print(f"  Packages needing types.py: {report['summary']['packages_needing_types_file']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
