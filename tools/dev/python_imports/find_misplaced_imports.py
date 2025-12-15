#!/usr/bin/env python3
import ast
import os
import subprocess
import sys
from collections import defaultdict

from tqdm import tqdm

# === Config ===
ROOT_DIR = "./metta"
OUT_PATH = "./tools/dev/python_imports/misplaced_imports_report.txt"
SKIP_DIRS = {".venv", ".bazel_output"}

# Cache for measured import times
import_timings = {}

# for measure_import_time, specifically


def measure_import_time(import_stmt: str) -> int:
    cmd = [
        sys.executable,
        "-c",
        (f"import time; start=time.time(); {import_stmt}; print(int((time.time()-start)*1000))"),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except Exception:
        return 0


def extract_module_names(node):
    """Return the module names from an Import or ImportFrom node"""
    names = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            names.add(alias.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            names.add(node.module)
    return names


misplaced_imports = defaultdict(list)
embedded_imports = defaultdict(list)


def visit_file(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    try:
        tree = ast.parse(src, filename=path)
    except Exception:
        return

    # Track top-level imports, ignoring blank lines and comments
    top_imports_done = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if top_imports_done:
                misplaced_imports[path].append((ast.get_source_segment(src, node), node))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            # Docstring at the top of the file – ignore
            continue
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node.value.value.strip() == "":
            # Blank line – ignore
            continue
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node.value.value.strip().startswith("#")
        ):
            # Comment – ignore
            continue
        else:
            top_imports_done = True

    # Find embedded imports
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Skip if it's at top level
            if node in [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]:
                continue
            embedded_imports[path].append((ast.get_source_segment(src, node), node))


def collect_imports():
    py_files = []
    for root, dirs, files in os.walk(ROOT_DIR):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))

    # Progress bar for scanning files
    for path in tqdm(py_files, desc="Scanning Python files...", unit="file"):
        visit_file(path)


def timing_prefix(modname: str) -> str:
    if not modname:
        return "[?ms] "
    # Skip modules already imported by Python
    if modname in sys.modules:
        import_timings[modname] = 0
    # Only measure once
    if modname not in import_timings:
        import_timings[modname] = measure_import_time(modname)
    return f"[{import_timings[modname]}ms] "


def write_report():
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        out.write("=== Imports not at top of file ===\n\n")
        for path, items in tqdm(
            misplaced_imports.items(),
            desc="Writing misplaced imports...",
            unit="file",
        ):
            out.write(f"{path}\n")
            for text, _ in items:
                timing_ms = import_timings.get(text.strip(), 0)
                prefix = f"[{timing_ms}ms] "
                out.write(f"  {prefix}{text}\n")

            out.write("\n")

        out.write("\n=== Imports inside classes or functions ===\n\n")
        for path, items in tqdm(
            embedded_imports.items(),
            desc="Writing embedded imports...",
            unit="file",
        ):
            out.write(f"{path}\n")
            for text, _ in items:
                timing_ms = import_timings.get(text.strip(), 0)
                prefix = f"[{timing_ms}ms] "
                out.write(f"  {prefix}{text}\n")

            out.write("\n")


if __name__ == "__main__":
    collect_imports()

    # Collect unique import statements (full text)
    seen_import_stmts = set()
    for items in list(misplaced_imports.values()) + list(embedded_imports.values()):
        for text, _ in items:
            seen_import_stmts.add(text.strip())

    # Make project root importable
    sys.path.insert(0, os.path.abspath(ROOT_DIR))

    # Measure each unique import statement exactly once
    for stmt in tqdm(sorted(seen_import_stmts), desc="Measuring import times"):
        if stmt not in import_timings:
            import_timings[stmt] = measure_import_time(stmt)

    write_report()
    print(f"Report written to {OUT_PATH}")
