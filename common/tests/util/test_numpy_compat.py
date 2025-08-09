"""Tests for metta.common.util.numpy_compat module."""

import importlib
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestNumpyCompat:
    """Test cases for the numpy_compat module compatibility shims."""

    def setup_method(self):
        """Reset numpy attributes before each test."""
        # Store original attributes to restore later
        self.original_attrs = {}
        self.test_attrs = [
            "byte", "short", "intc", "int_", "longlong",
            "ubyte", "ushort", "uintc", "uint", "ulonglong",
            "float_", "double", "longdouble",
            "csingle", "cdouble", "clongdouble"
        ]

        for attr in self.test_attrs:
            if hasattr(np, attr):
                self.original_attrs[attr] = getattr(np, attr)

    def teardown_method(self):
        """Restore original numpy attributes after each test."""
        # Remove any attributes we may have added
        for attr in self.test_attrs:
            if hasattr(np, attr) and attr not in self.original_attrs:
                delattr(np, attr)

        # Restore original attributes
        for attr, value in self.original_attrs.items():
            setattr(np, attr, value)

    def test_integer_type_compatibility(self):
        """Test that integer type aliases are added when missing."""
        # Remove the attributes to simulate NumPy 2.0
        attrs_to_test = ["byte", "short", "intc", "int_", "longlong"]
        for attr in attrs_to_test:
            if hasattr(np, attr):
                delattr(np, attr)

        # Re-import the module to trigger the compatibility code
        import metta.common.util.numpy_compat
        importlib.reload(metta.common.util.numpy_compat)

        # Check that the attributes were added back
        assert hasattr(np, "byte")
        assert np.byte == np.int8

        assert hasattr(np, "short")
        assert np.short == np.int16

        assert hasattr(np, "intc")
        assert np.intc == np.int32

        assert hasattr(np, "int_")
        assert np.int_ == np.int64

        assert hasattr(np, "longlong")
        assert np.longlong == np.int64

    def test_unsigned_integer_type_compatibility(self):
        """Test that unsigned integer type aliases are added when missing."""
        attrs_to_test = ["ubyte", "ushort", "uintc", "uint", "ulonglong"]
        for attr in attrs_to_test:
            if hasattr(np, attr):
                delattr(np, attr)

        # Re-import the module
        import metta.common.util.numpy_compat
        importlib.reload(metta.common.util.numpy_compat)

        # Check that the attributes were added back
        assert hasattr(np, "ubyte")
        assert np.ubyte == np.uint8

        assert hasattr(np, "ushort")
        assert np.ushort == np.uint16

        assert hasattr(np, "uintc")
        assert np.uintc == np.uint32

        assert hasattr(np, "uint")
        assert np.uint == np.uint64

        assert hasattr(np, "ulonglong")
        assert np.ulonglong == np.uint64

    def test_float_type_compatibility(self):
        """Test that float type aliases are added when missing."""
        attrs_to_test = ["float_", "double"]
        for attr in attrs_to_test:
            if hasattr(np, attr):
                delattr(np, attr)

        # Re-import the module
        import metta.common.util.numpy_compat
        importlib.reload(metta.common.util.numpy_compat)

        # Check that the attributes were added back
        assert hasattr(np, "float_")
        assert np.float_ == np.float64

        assert hasattr(np, "double")
        assert np.double == np.float64

    def test_longdouble_compatibility_with_float128(self):
        """Test longdouble compatibility when float128 is available."""
        if hasattr(np, "longdouble"):
            delattr(np, "longdouble")

        # Mock float128 availability
        with patch.object(np, "float128", np.float64, create=True):
            # Re-import the module
            import metta.common.util.numpy_compat
            importlib.reload(metta.common.util.numpy_compat)

            assert hasattr(np, "longdouble")
            assert np.longdouble == np.float64  # Since we mocked float128 as float64

    def test_longdouble_compatibility_without_float128(self):
        """Test longdouble compatibility when float128 is not available."""
        if hasattr(np, "longdouble"):
            delattr(np, "longdouble")

        # Ensure float128 is not available
        original_float128 = getattr(np, "float128", None)
        if hasattr(np, "float128"):
            delattr(np, "float128")

        try:
            # Re-import the module
            import metta.common.util.numpy_compat
            importlib.reload(metta.common.util.numpy_compat)

            assert hasattr(np, "longdouble")
            assert np.longdouble == np.float64  # Fallback when float128 not available
        finally:
            # Restore float128 if it existed
            if original_float128 is not None:
                np.float128 = original_float128

    def test_complex_type_compatibility(self):
        """Test that complex type aliases are added when missing."""
        attrs_to_test = ["csingle", "cdouble"]
        for attr in attrs_to_test:
            if hasattr(np, attr):
                delattr(np, attr)

        # Re-import the module
        import metta.common.util.numpy_compat
        importlib.reload(metta.common.util.numpy_compat)

        # Check that the attributes were added back
        assert hasattr(np, "csingle")
        assert np.csingle == np.complex64

        assert hasattr(np, "cdouble")
        assert np.cdouble == np.complex128

    def test_clongdouble_compatibility_with_complex256(self):
        """Test clongdouble compatibility when complex256 is available."""
        if hasattr(np, "clongdouble"):
            delattr(np, "clongdouble")

        # Mock complex256 availability
        with patch.object(np, "complex256", np.complex128, create=True):
            # Re-import the module
            import metta.common.util.numpy_compat
            importlib.reload(metta.common.util.numpy_compat)

            assert hasattr(np, "clongdouble")
            assert np.clongdouble == np.complex128  # Since we mocked complex256 as complex128

    def test_clongdouble_compatibility_without_complex256(self):
        """Test clongdouble compatibility when complex256 is not available."""
        if hasattr(np, "clongdouble"):
            delattr(np, "clongdouble")

        # Ensure complex256 is not available
        original_complex256 = getattr(np, "complex256", None)
        if hasattr(np, "complex256"):
            delattr(np, "complex256")

        try:
            # Re-import the module
            import metta.common.util.numpy_compat
            importlib.reload(metta.common.util.numpy_compat)

            assert hasattr(np, "clongdouble")
            assert np.clongdouble == np.complex128  # Fallback when complex256 not available
        finally:
            # Restore complex256 if it existed
            if original_complex256 is not None:
                np.complex256 = original_complex256

    def test_existing_attributes_not_overwritten(self):
        """Test that existing NumPy attributes are not overwritten."""
        # Ensure some attributes exist with custom values
        original_byte = getattr(np, "byte", None)
        original_short = getattr(np, "short", None)

        # Set custom values
        np.byte = "custom_byte_value"
        np.short = "custom_short_value"

        try:
            # Re-import the module
            import metta.common.util.numpy_compat
            importlib.reload(metta.common.util.numpy_compat)

            # Check that our custom values were not overwritten
            assert np.byte == "custom_byte_value"
            assert np.short == "custom_short_value"
        finally:
            # Restore original values
            if original_byte is not None:
                np.byte = original_byte
            elif hasattr(np, "byte"):
                delattr(np, "byte")

            if original_short is not None:
                np.short = original_short
            elif hasattr(np, "short"):
                delattr(np, "short")

    def test_module_imports_successfully(self):
        """Test that the module can be imported without errors."""
        # This test ensures the module-level code executes successfully
        try:
            import metta.common.util.numpy_compat
            importlib.reload(metta.common.util.numpy_compat)
        except Exception as e:
            pytest.fail(f"Module import failed: {e}")

    def test_wandb_compatibility_attributes_present(self):
        """Test that key attributes needed for WandB compatibility are present."""
        # Import the module to ensure compatibility attributes are set
        import metta.common.util.numpy_compat
        importlib.reload(metta.common.util.numpy_compat)

        # These are the key attributes that WandB expects
        wandb_required_attrs = [
            "byte", "short", "intc", "int_", "longlong",
            "ubyte", "ushort", "uintc", "uint", "ulonglong",
            "float_", "double", "csingle", "cdouble"
        ]

        for attr in wandb_required_attrs:
            assert hasattr(np, attr), f"Required WandB attribute {attr} not present"
            # Ensure the attribute is callable/usable as a type
            attr_value = getattr(np, attr)
            assert callable(attr_value) or hasattr(attr_value, "dtype"), f"Attribute {attr} is not a valid NumPy type"

    def test_compatibility_attributes_work_correctly(self):
        """Test that compatibility attributes function as valid NumPy types."""
        # Import the module to ensure compatibility attributes are set
        import metta.common.util.numpy_compat
        importlib.reload(metta.common.util.numpy_compat)

        # Test core attributes that are essential for WandB compatibility
        essential_attrs = ["byte", "short", "intc", "ubyte", "float_", "csingle"]

        for attr_name in essential_attrs:
            if hasattr(np, attr_name):
                attr_value = getattr(np, attr_name)
                assert callable(attr_value), f"{attr_name} should be callable"

                # Test that it can create arrays
                try:
                    # Use appropriate test values for each type
                    if attr_name in ['csingle', 'cdouble', 'complex64', 'complex128']:
                        test_val = 1+0j  # Complex types need complex values
                    else:
                        test_val = 1  # All other types can use simple integer
                    arr = np.array([test_val], dtype=attr_value)
                    assert isinstance(arr, np.ndarray)
                except (TypeError, ValueError) as e:
                    pytest.fail(f"{attr_name} failed to create array: {e}")

        # Test that special types have correct kinds
        if hasattr(np, "longlong"):
            arr = np.array([1], dtype=np.longlong)
            assert arr.dtype.kind in ['i', 'u'], "longlong should be integer type"

    def test_types_can_create_arrays(self):
        """Test that the compatibility types can actually create NumPy arrays."""
        # Import the module
        import metta.common.util.numpy_compat
        importlib.reload(metta.common.util.numpy_compat)

        # Test that we can create arrays with these types
        test_value = 42

        if hasattr(np, "byte"):
            arr = np.array([test_value], dtype=np.byte)
            assert arr.dtype == np.int8

        if hasattr(np, "ubyte"):
            arr = np.array([test_value], dtype=np.ubyte)
            assert arr.dtype == np.uint8

        if hasattr(np, "float_"):
            arr = np.array([3.14], dtype=np.float_)
            assert arr.dtype == np.float64

        if hasattr(np, "csingle"):
            arr = np.array([1+2j], dtype=np.csingle)
            assert arr.dtype == np.complex64

    def test_wandb_compatibility_integration(self):
        """Test that numpy compatibility works in a WandB-like scenario."""
        import metta.common.util.numpy_compat

        # Simulate what WandB might do with these deprecated types
        try:
            # This is the actual use case - WandB using deprecated numpy attributes
            test_data = {
                'byte_data': np.array([1, 2, 3], dtype=np.byte),
                'ubyte_data': np.array([1, 2, 3], dtype=np.ubyte),
                'float_data': np.array([1.0, 2.0], dtype=np.float_),
                'complex_data': np.array([1+2j], dtype=np.csingle),
            }

            # All should create valid arrays
            for name, arr in test_data.items():
                assert isinstance(arr, np.ndarray), f"{name} failed to create array"
                assert arr.size > 0, f"{name} created empty array"

        except AttributeError as e:
            pytest.fail(f"WandB compatibility failed - missing attribute: {e}")
        except Exception as e:
            pytest.fail(f"WandB compatibility failed: {e}")

    def test_module_documentation(self):
        """Test that the module has proper documentation."""
        import metta.common.util.numpy_compat

        # Check that the module has a docstring explaining its purpose
        assert metta.common.util.numpy_compat.__doc__ is not None
        assert "NumPy 2.0 compatibility" in metta.common.util.numpy_compat.__doc__
        assert "WandB" in metta.common.util.numpy_compat.__doc__
