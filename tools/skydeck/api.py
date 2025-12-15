#!/usr/bin/env -S uv run
"""Skydeck API - provides schema extraction and tool discovery for external tools.

Usage:
    # Get schema for a tool
    uv run ./tools/skydeck/api.py schema arena.train

    # List all known train tools
    uv run ./tools/skydeck/api.py tools
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from metta.common.tool.schema import extract_schema

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

# Tool function names to look for in recipe modules
TOOL_NAMES = ["train", "evaluate", "play", "replay"]


def load_class_or_factory(import_path: str) -> type[BaseModel] | Callable[[], BaseModel] | None:
    """Load a class or factory function from an import path."""
    if "." not in import_path:
        return None
    module_path, symbol = import_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, symbol, None)
    except (ImportError, AttributeError):
        return None


def resolve_to_model_class(import_path: str) -> type[BaseModel] | None:
    """Resolve an import path to a Pydantic model class."""
    obj = load_class_or_factory(import_path)

    if obj is None:
        try:
            from metta.common.tool.tool_path import resolve_and_load_tool_maker

            obj = resolve_and_load_tool_maker(import_path)
        except ImportError:
            pass

    if obj is None:
        return None

    if inspect.isclass(obj) and issubclass(obj, BaseModel):
        return obj

    if callable(obj):
        try:
            result = obj()
            if isinstance(result, BaseModel):
                return type(result)
        except Exception:
            pass

    return None


def get_schema(import_path: str) -> dict[str, Any]:
    """Get schema for a tool/config path. Returns {"schema": ...} or {"invalid_keys": [...]}."""
    output: dict[str, Any] = {}
    try:
        model_class = resolve_to_model_class(import_path)
        if model_class is not None:
            output["schema"] = extract_schema(model_class)
        else:
            output["invalid_keys"] = [import_path]
    except Exception:
        output["invalid_keys"] = [import_path]

    if not output:
        output = {"schema": {}}
    return output


def discover_recipe_tools() -> dict[str, str]:
    """Discover all tool functions in recipes modules.

    Scans recipes.experiment and recipes.prod for modules containing
    train, evaluate, play, or replay functions.

    Returns:
        Dict mapping tool paths to their Tool class name, e.g.:
        {"recipes.experiment.arena.train": "TrainTool", ...}
    """
    tools: dict[str, str] = {}

    # Find the recipes package
    try:
        import recipes

        recipes_path = Path(recipes.__file__).parent
    except ImportError:
        return tools

    # Directories to scan
    scan_dirs = [
        ("recipes.experiment", recipes_path / "experiment"),
        ("recipes.prod", recipes_path / "prod"),
    ]

    for base_module, base_path in scan_dirs:
        if not base_path.exists():
            continue

        # Walk through all Python files
        for py_file in base_path.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            # Convert file path to module path
            rel_path = py_file.relative_to(base_path)
            parts = list(rel_path.parts)
            parts[-1] = parts[-1][:-3]  # Remove .py

            module_path = f"{base_module}.{'.'.join(parts)}"

            # Try to import and check for tool functions
            try:
                module = importlib.import_module(module_path)
                for tool_name in TOOL_NAMES:
                    if hasattr(module, tool_name):
                        func = getattr(module, tool_name)
                        if callable(func):
                            tool_path = f"{module_path}.{tool_name}"
                            # Try to get the return type
                            try:
                                result = func()
                                tool_type = type(result).__name__
                            except Exception:
                                tool_type = "Unknown"
                            tools[tool_path] = tool_type
            except Exception:
                # Skip modules that fail to import
                continue

    return dict(sorted(tools.items()))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage:", file=sys.stderr)
        print("  api.py schema <import_path>  - Get schema for a tool", file=sys.stderr)
        print("  api.py tools                 - List all known tools", file=sys.stderr)
        return 1

    command = sys.argv[1]

    if command == "schema":
        if len(sys.argv) < 3:
            print("Usage: api.py schema <import_path>", file=sys.stderr)
            return 1
        import_path = sys.argv[2]
        output = get_schema(import_path)
        print(json.dumps(output, indent=2, default=str))

    elif command == "tools":
        tools = discover_recipe_tools()
        print(json.dumps({"tools": tools}, indent=2))

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
