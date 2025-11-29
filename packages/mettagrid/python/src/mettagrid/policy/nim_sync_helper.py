"""Helper module for agents.py files to sync Nim code with caching.

This module can be imported by agents.py files in submission archives
to ensure Nim code is synced and compiled, using cache when possible.
"""

from __future__ import annotations

from pathlib import Path


def sync_nim_code() -> None:
    """Sync and compile Nim code in the current directory, using cache when possible.

    This function is designed to be called from agents.py files in submission archives.
    It will:
    1. Detect the directory containing Nim source files
    2. Use cached compilation if available and source hasn't changed
    3. Sync and compile only if needed

    Raises:
        RuntimeError: If sync or compilation fails
    """
    try:
        from mettagrid.policy.nim_build_cache import ensure_nim_compiled
    except ImportError:
        # Fallback: try to import from cogames if mettagrid not available
        try:
            import importlib.util
            import sys

            # Try to find nim_build_cache in cogames package
            cogames_path = None
            for path in sys.path:
                candidate = Path(path) / "cogames" / "policy" / "nim_agents"
                if candidate.exists():
                    cogames_path = candidate.parent.parent.parent
                    break

            if cogames_path:
                spec = importlib.util.spec_from_file_location(
                    "nim_build_cache",
                    cogames_path / "mettagrid" / "policy" / "nim_build_cache.py",
                )
                if spec and spec.loader:
                    nim_build_cache = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(nim_build_cache)
                    ensure_nim_compiled = nim_build_cache.ensure_nim_compiled
                else:
                    raise ImportError("Could not load nim_build_cache")
            else:
                raise ImportError("nim_build_cache not found")
        except Exception:
            raise RuntimeError(
                "Failed to import nim_build_cache. Nim compilation caching requires mettagrid package."
            ) from None

    # Find the directory containing this file (agents.py)
    # Walk up to find the directory with Nim files
    current_file = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
    if not current_file.exists():
        # Fallback: use current working directory
        current_file = Path.cwd()

    # Try to find the directory with Nim source files
    # Check current directory and parent directories
    search_dirs = [current_file.parent, current_file.parent.parent]
    if "__file__" in globals():
        # If called from agents.py, use that file's directory
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_file = frame.f_back.f_globals.get("__file__")
            if caller_file:
                search_dirs.insert(0, Path(caller_file).parent)

    nim_dir = None
    for search_dir in search_dirs:
        # Look for indicators of Nim code
        if (search_dir / "nimby.lock").exists() or list(search_dir.glob("*.nim")):
            nim_dir = search_dir
            break

    if nim_dir is None:
        # If no Nim directory found, try current working directory
        cwd = Path.cwd()
        if (cwd / "nimby.lock").exists() or list(cwd.glob("*.nim")):
            nim_dir = cwd
        else:
            raise FileNotFoundError(f"No Nim source files found. Searched: {search_dirs}")

    # Ensure Nim is compiled (will use cache if available)
    ensure_nim_compiled(nim_dir)
