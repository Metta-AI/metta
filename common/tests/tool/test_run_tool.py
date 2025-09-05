import os
import subprocess

import pytest


@pytest.fixture
def with_extra_imports_root(monkeypatch):
    extra_imports_root = os.path.join(os.path.dirname(__file__), "fixtures/extra-import-root")
    monkeypatch.setenv("PYTHONPATH", extra_imports_root)


def test_basic(with_extra_imports_root):
    result = subprocess.check_output(["tool", "mypackage.tools.TestTool"], text=True)
    assert "TestTool invoked" in result


def test_unknown_tool(with_extra_imports_root):
    result = subprocess.run(["tool", "mypackage.tools.NoSuchTool"], text=True, capture_output=True)
    assert result.returncode > 0
    assert "module 'mypackage.tools' has no attribute 'NoSuchTool'" in result.stderr


def test_unknown_module(with_extra_imports_root):
    result = subprocess.run(["tool", "mypackage.no_such_tools.TestTool"], text=True, capture_output=True)
    assert result.returncode > 0
    assert "No module named 'mypackage.no_such_tools'" in result.stderr
