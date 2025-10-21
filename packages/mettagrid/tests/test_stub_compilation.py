"""Ensure critical type stubs remain syntactically valid."""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest


def test_mettagrid_c_stub_compiles() -> None:
    """mettagrid_c.pyi should always parse so type checkers stay usable."""

    repo_root = Path(__file__).resolve().parents[3]
    stub_path = repo_root / "packages" / "mettagrid" / "python" / "src" / "mettagrid" / "mettagrid_c.pyi"

    try:
        py_compile.compile(str(stub_path), doraise=True)
    except Exception as exc:  # pragma: no cover - failure path
        pytest.fail(f"Failed to compile mettagrid_c.pyi: {exc}")
