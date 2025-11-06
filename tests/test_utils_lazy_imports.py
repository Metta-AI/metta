"""Test lazy imports in metta.utils package."""



def test_lazy_import_file_module() -> None:
    """Test that file module can be lazily imported from metta.utils."""
    import metta.utils

    # Verify the module was imported
    assert metta.utils.file is not None
    assert hasattr(metta.utils.file, "__name__")


def test_lazy_import_uri_module() -> None:
    """Test that uri module can be lazily imported from metta.utils."""

    # Verify the module was imported
    assert metta.utils.uri is not None
    assert hasattr(metta.utils.uri, "__name__")


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
