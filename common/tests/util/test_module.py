import importlib
import os

import pytest

from mettagrid.util.module import load_symbol


@pytest.fixture
def with_extra_imports_root(monkeypatch):
    """Add test fixtures to Python path for recipe/tool discovery."""
    extra_imports_root = os.path.join(os.path.dirname(__file__), "fixtures/extra-import-root")
    monkeypatch.setenv("PYTHONPATH", extra_imports_root)
    monkeypatch.syspath_prepend(extra_imports_root)


def test_load_symbol_with_builtin() -> None:
    result = load_symbol("builtins.str")
    assert result is str


def test_load_symbol_with_stdlib_function() -> None:
    result = load_symbol("importlib.import_module")
    assert result is importlib.import_module


def test_load_symbol_invalid_format_raises_value_error() -> None:
    with pytest.raises(ModuleNotFoundError) as excinfo:
        load_symbol("NotFullyQualifiedName")
    assert "Invalid symbol name" in str(excinfo.value)


def test_load_symbol_missing_module_raises_module_not_found_error() -> None:
    with pytest.raises(ModuleNotFoundError):
        load_symbol("this_module_does_not_exist__abcdef.Symbol")


def test_load_self() -> None:
    result = load_symbol("mettagrid.util.module.load_symbol")
    assert callable(result)


def test_load_config() -> None:
    result = load_symbol("mettagrid.config.Config")
    assert isinstance(result, type)
    assert result.__name__ == "Config"


def test_load_nested_symbol(with_extra_imports_root) -> None:
    result = load_symbol("foo.bar.baz.Foo.Bar.Baz")
    assert result.__name__ == "Baz"
