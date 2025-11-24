"""Test lazy imports in metta.utils package."""

from __future__ import annotations


def test_lazy_import_file_module() -> None:
    """Test that file module can be lazily imported from metta.utils."""
    from metta.utils import file

    # Verify the module was imported (now redirects to metta.common.util.file)
    assert file is not None
    assert hasattr(file, "__name__")
    # Verify it's actually the file module from common.util
    assert file.__name__ == "metta.common.util.file"


def test_lazy_import_uri_module() -> None:
    """Test that uri module can be lazily imported from metta.utils."""
    from metta.utils import uri

    # Verify the module was imported
    assert uri is not None
    assert hasattr(uri, "__name__")


def test_import_star_includes_file_and_uri() -> None:
    """Test that __all__ properly exports file and uri modules."""
    import metta.utils

    # Check __all__ declares both modules
    assert "file" in metta.utils.__all__
    assert "uri" in metta.utils.__all__


def test_invalid_attribute_raises_error() -> None:
    """Test that accessing non-existent attributes raises AttributeError."""
    import pytest

    import metta.utils

    with pytest.raises(AttributeError, match="'metta.utils'.*nonexistent_module"):
        _ = metta.utils.nonexistent_module  # type: ignore[attr-defined]
