import subprocess
import sys
import tomllib

import pytest

from metta.common.util.fs import get_repo_root

pytestmark = pytest.mark.setup

REPO_ROOT = get_repo_root()

# Directories to exclude from pyproject.toml discovery
EXCLUDE_DIRS = {".bazel_output", ".venv", "node_modules"}

# Entry points that legitimately need heavy imports at startup.
EXCLUDE_ENTRY_POINTS = {
    "cogames",  # Game runner, needs torch for policy inference
    "mettagrid-demo",  # Visualization demo, needs torch at startup
}

HEAVY_IMPORTS = ["torch"]


def discover_entry_points() -> list[tuple[str, str]]:
    """Discover all entry points from pyproject.toml files in the repo.

    Returns a list of (script_name, module_path) tuples.
    """
    entry_points = []

    for pyproject_path in REPO_ROOT.rglob("pyproject.toml"):
        # Skip excluded directories
        if any(excluded in pyproject_path.parts for excluded in EXCLUDE_DIRS):
            continue

        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            continue

        scripts = data.get("project", {}).get("scripts", {})
        for script_name, entry_point in scripts.items():
            # Entry point format: "module.path:function"
            module_path = entry_point.split(":")[0]
            entry_points.append((script_name, module_path))

    return entry_points


def discover_codebase_prefixes() -> tuple[str, ...]:
    """Discover package prefixes from pyproject.toml files in the repo."""
    prefixes = set()

    for pyproject_path in REPO_ROOT.rglob("pyproject.toml"):
        if any(excluded in pyproject_path.parts for excluded in EXCLUDE_DIRS):
            continue

        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            continue

        # Get package name from [project].name
        name = data.get("project", {}).get("name")
        if name:
            # Convert package name to module prefix (e.g., "my-package" -> "my_package.")
            prefixes.add(f"{name.replace('-', '_')}.")

    return tuple(prefixes)


ENTRY_POINTS = discover_entry_points()
CODEBASE_PREFIXES = discover_codebase_prefixes()


@pytest.mark.parametrize("script_name,module", ENTRY_POINTS, ids=[ep[0] for ep in ENTRY_POINTS])
def test_mains_no_heavy_imports(script_name: str, module: str) -> None:
    """Ensure main entry points do not import heavy modules at startup."""
    if script_name in EXCLUDE_ENTRY_POINTS:
        pytest.skip(f"{script_name} is in EXCLUDE_ENTRY_POINTS (legitimately needs heavy imports)")

    cmd = [sys.executable, "-X", "importtime", "-c", f"import {module}"]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        pytest.skip(f"Module {module} failed to import: {completed.stderr}")

    def parse_module_name(line: str) -> str | None:
        """Extract module name from importtime output line."""
        # Format: "import time: <self> | <cumulative> | <indent><module_name>"
        if "|" not in line:
            return None
        parts = line.split("|")
        if len(parts) < 3:
            return None
        return parts[-1].strip()

    def is_heavy_module(module_name: str | None, heavy: str) -> bool:
        """Check if module is the heavy import or a submodule of it."""
        if not module_name:
            return False
        return module_name == heavy or module_name.startswith(f"{heavy}.")

    def is_codebase_module(module_name: str | None) -> bool:
        """Check if module belongs to this codebase."""
        if not module_name:
            return False
        return any(module_name.startswith(prefix) for prefix in CODEBASE_PREFIXES)

    for heavy in HEAVY_IMPORTS:
        lines = completed.stderr.splitlines()

        # Check if heavy module was imported at all
        if not any(is_heavy_module(parse_module_name(line), heavy) for line in lines):
            continue

        # Find the codebase module that triggered the heavy import.
        # importtime prints modules when they FINISH (deepest first).
        # So the importer appears AFTER the heavy module in the output.
        codebase_imports = []

        # Find the last actual heavy module line
        last_heavy_idx = -1
        for i, line in enumerate(lines):
            module_name = parse_module_name(line)
            if is_heavy_module(module_name, heavy):
                last_heavy_idx = i

        # Collect codebase modules and find the first one after heavy
        torch_trigger = None
        for i, line in enumerate(lines):
            module_name = parse_module_name(line)
            if is_codebase_module(module_name):
                codebase_imports.append(line)
                if i > last_heavy_idx and torch_trigger is None:
                    torch_trigger = line

        source_hint = ""
        if torch_trigger:
            source_hint = f"\nSource (codebase module that imports {heavy}):\n{torch_trigger}\n"

        pytest.fail(
            f"{heavy} imported by {script_name} ({module}).\n\n"
            f"Import chain (codebase modules only):\n" + "\n".join(codebase_imports) + source_hint + "\n"
            f"Heavy imports like {heavy} should be lazy-loaded to keep non-execution "
            f"CLI paths (--help, --list, --dry-run) fast. Move the import inside the "
            f"function that needs it.\n\n"
            f"If {script_name} legitimately requires {heavy} at startup, add it to "
            f"EXCLUDE_ENTRY_POINTS in tests/setup/test_mains_for_heavy_imports.py."
        )
