"""Test lazy imports in metta.utils package."""

import pytest

import metta.utils


def test_lazy_import_file_module() -> None:
    """Test that file module can be lazily imported from metta.utils."""

    # Verify the module was imported (redirects to mettagrid.util.file)
    assert metta.utils.file is not None
    assert hasattr(metta.utils.file, "__name__")
    # Verify it's actually the file module from mettagrid
    assert metta.utils.file.__name__ == "mettagrid.util.file"


def test_import_star_includes_file() -> None:
    """Test that __all__ properly exports file module."""

    # Check __all__ declares file module
    assert "file" in metta.utils.__all__


def test_invalid_attribute_raises_error() -> None:
    """Test that accessing non-existent attributes raises AttributeError."""

    with pytest.raises(AttributeError, match="'metta.utils'.*nonexistent_module"):
        _ = metta.utils.nonexistent_module
