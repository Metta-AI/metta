from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

ALLOWED_METTA_PACKAGES = ["mettagrid"]

EXCLUDE_FILES = set()


def find_forbidden_imports(file_path: Path) -> list[ast.stmt]:
    """Return list of AST nodes with forbidden imports in a file.

    Flags both:
    - import metta.rl[.x]
    - from metta.rl[.x] import ...
    """
    source = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        # Skip files with syntax errors
        return []

    bad_nodes: list[ast.stmt] = []
    for node in ast.walk(tree):
        modules: list[str] = []

        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # Only care about absolute imports (level == 0)
            if node.level == 0 and node.module is not None:
                modules.append(node.module)
        else:
            continue

        for module in modules:
            if module.startswith("metta.") and not any(
                module.startswith(allowed) for allowed in ALLOWED_METTA_PACKAGES
            ):
                bad_nodes.append(node)

    return bad_nodes


def iter_python_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.py")


def test_no_forbidden_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "python" / "src"
    assert src_dir.is_dir(), f"Expected source directory not found: {src_dir}"

    failures: list[str] = []
    for py_file in iter_python_files(src_dir):
        if str(py_file.relative_to(src_dir)) in EXCLUDE_FILES:
            continue
        for node in find_forbidden_imports(py_file):
            rel = py_file.relative_to(src_dir)
            failures.append(f"{rel}:{node.lineno}: {ast.unparse(node)}")

    if failures:
        details = "\n".join(sorted(failures))
        raise AssertionError("Forbidden imports detected:\n" + details)
