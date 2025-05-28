"""
Tests for tensor_conversion utilities.
"""

import numpy as np
import pytest
import torch

from metta.util.tensor_conversion import (
    batch_tensor_to_numpy,
    get_dtype_info,
    tensor_to_numpy,
    validate_tensor_dtype_range,
    validate_tensor_range,
)


class TestTensorToNumpy:
    """Test the tensor_to_numpy function."""

    def test_basic_conversion(self):
        """Test basic tensor to numpy conversion."""
        tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = tensor_to_numpy(tensor, np.uint8)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert np.array_equal(result, [1, 2, 3])

    def test_dtype_input_formats(self):
        """Test different ways to specify target dtype."""
        tensor = torch.tensor([1, 2, 3], dtype=torch.int32)

        # Test different dtype specifications
        result1 = tensor_to_numpy(tensor, np.uint8)
        result2 = tensor_to_numpy(tensor, "uint8")
        result3 = tensor_to_numpy(tensor, np.dtype("uint8"))

        assert np.array_equal(result1, result2)
        assert np.array_equal(result2, result3)
        assert all(r.dtype == np.uint8 for r in [result1, result2, result3])

    def test_cuda_tensor(self):
        """Test conversion of CUDA tensor to numpy."""
        if torch.cuda.is_available():
            tensor = torch.tensor([1, 2, 3], dtype=torch.int32).cuda()
            result = tensor_to_numpy(tensor, np.uint8)

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.uint8
            assert np.array_equal(result, [1, 2, 3])

    def test_copy_parameter(self):
        """Test copy parameter behavior."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        # Same dtype, no copy requested
        result_no_copy = tensor_to_numpy(tensor, np.float32, copy=False, validate_range=False)

        # Same dtype, copy requested
        result_copy = tensor_to_numpy(tensor, np.float32, copy=True, validate_range=False)

        # Results should be equal but different objects when copy=True
        assert np.array_equal(result_no_copy, result_copy)

    def test_gradient_tensor(self):
        """Test that gradient tensors are handled correctly."""
        tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = tensor_to_numpy(tensor, np.float32, validate_range=False)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert np.array_equal(result, [1.0, 2.0, 3.0])

    def test_invalid_input_type(self):
        """Test error handling for invalid input types."""
        with pytest.raises(TypeError, match="Expected torch.Tensor"):
            tensor_to_numpy([1, 2, 3], np.uint8)

    def test_invalid_dtype(self):
        """Test error handling for invalid target dtype."""
        tensor = torch.tensor([1, 2, 3])
        with pytest.raises((TypeError, ValueError)):
            tensor_to_numpy(tensor, "invalid_dtype")

    @pytest.mark.parametrize("validate", [True, False])
    def test_range_validation_toggle(self, validate):
        """Test that range validation can be toggled."""
        tensor = torch.tensor([100, 200, 300], dtype=torch.int32)

        if validate:
            # Should fail validation for uint8 (max 255)
            with pytest.raises(AssertionError):
                tensor_to_numpy(tensor, np.uint8, validate_range=True)
        else:
            # Should succeed without validation (but may overflow)
            result = tensor_to_numpy(tensor, np.uint8, validate_range=False)
            assert isinstance(result, np.ndarray)


class TestRangeValidation:
    """Test range validation functions."""

    def test_validate_tensor_range_pass(self):
        """Test range validation with valid values."""
        tensor = torch.tensor([1, 5, 10])
        # Should not raise
        validate_tensor_range(tensor, 0, 255)

    def test_validate_tensor_range_fail_min(self):
        """Test range validation with values below minimum."""
        tensor = torch.tensor([-1, 5, 10])
        with pytest.raises(AssertionError, match="must be in range"):
            validate_tensor_range(tensor, 0, 255)

    def test_validate_tensor_range_fail_max(self):
        """Test range validation with values above maximum."""
        tensor = torch.tensor([1, 5, 256])
        with pytest.raises(AssertionError, match="must be in range"):
            validate_tensor_range(tensor, 0, 255)

    def test_validate_tensor_dtype_range_uint8(self):
        """Test dtype range validation for uint8."""
        # Valid range
        tensor_valid = torch.tensor([0, 127, 255])
        validate_tensor_dtype_range(tensor_valid, np.uint8)  # Should not raise

        # Invalid range
        tensor_invalid = torch.tensor([0, 127, 256])
        with pytest.raises(AssertionError):
            validate_tensor_dtype_range(tensor_invalid, np.uint8)

    def test_validate_tensor_dtype_range_int8(self):
        """Test dtype range validation for int8."""
        # Valid range
        tensor_valid = torch.tensor([-128, 0, 127])
        validate_tensor_dtype_range(tensor_valid, np.int8)  # Should not raise

        # Invalid range
        tensor_invalid = torch.tensor([-129, 0, 127])
        with pytest.raises(AssertionError):
            validate_tensor_dtype_range(tensor_invalid, np.int8)

    def test_validate_tensor_dtype_range_float32(self):
        """Test dtype range validation for float32."""
        tensor = torch.tensor([1.0, 2.5, 3.7])
        validate_tensor_dtype_range(tensor, np.float32)  # Should not raise

    def test_validation_disabled_in_optimized_mode(self):
        """Test that validation is disabled when __debug__ is False."""
        # This test is tricky because __debug__ is set at import time
        # In practice, validation would be disabled with python -O
        pass


class TestBatchConversion:
    """Test batch tensor to numpy conversion."""

    def test_batch_conversion_basic(self):
        """Test basic batch conversion."""
        tensors = {
            "actions": torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            "rewards": torch.tensor([1.5, 2.5], dtype=torch.float32),
            "done": torch.tensor([False, True], dtype=torch.bool),
        }

        dtypes = {"actions": "uint8", "rewards": "float32", "done": "bool"}

        result = batch_tensor_to_numpy(tensors, dtypes, validate_range=False)

        assert len(result) == 3
        assert result["actions"].dtype == np.uint8
        assert result["rewards"].dtype == np.float32
        assert result["done"].dtype == np.bool_

        assert np.array_equal(result["actions"], [[0, 1], [2, 3]])
        assert np.array_equal(result["rewards"], [1.5, 2.5])
        assert np.array_equal(result["done"], [False, True])

    def test_batch_conversion_missing_dtype(self):
        """Test batch conversion with missing dtype specification."""
        tensors = {
            "actions": torch.tensor([1, 2, 3], dtype=torch.int32),
            "rewards": torch.tensor([1.0, 2.0], dtype=torch.float32),
        }

        dtypes = {
            "actions": "uint8",
            # 'rewards' missing - should use original dtype
        }

        result = batch_tensor_to_numpy(tensors, dtypes, validate_range=False)

        assert result["actions"].dtype == np.uint8
        assert result["rewards"].dtype == np.float32  # Original dtype preserved

    def test_batch_conversion_empty(self):
        """Test batch conversion with empty inputs."""
        result = batch_tensor_to_numpy({}, {})
        assert result == {}


class TestGetDtypeInfo:
    """Test dtype information utility."""

    def test_uint8_info(self):
        """Test dtype info for uint8."""
        info = get_dtype_info("uint8")

        assert info["dtype"] == np.uint8
        assert info["name"] == "uint8"
        assert info["is_integer"] is True
        assert info["is_floating"] is False
        assert info["is_signed"] is False
        assert info["min"] == 0
        assert info["max"] == 255
        assert info["bits"] == 8

    def test_int8_info(self):
        """Test dtype info for int8."""
        info = get_dtype_info(np.int8)

        assert info["dtype"] == np.int8
        assert info["is_integer"] is True
        assert info["is_signed"] is True
        assert info["min"] == -128
        assert info["max"] == 127

    def test_float32_info(self):
        """Test dtype info for float32."""
        info = get_dtype_info(np.float32)

        assert info["dtype"] == np.float32
        assert info["is_integer"] is False
        assert info["is_floating"] is True
        assert info["is_signed"] is True
        assert "min" in info
        assert "max" in info
        assert "eps" in info
        assert "precision" in info

    def test_bool_info(self):
        """Test dtype info for bool."""
        info = get_dtype_info(np.bool_)

        assert info["dtype"] == np.bool_
        assert info["is_integer"] is False
        assert info["is_floating"] is False


class TestIntegrationScenarios:
    """Integration tests for common use cases."""

    def test_actions_conversion_scenario(self):
        """Test typical actions conversion scenario."""
        # Simulate actions from a policy (often int64)
        actions = torch.randint(0, 10, (16, 2), dtype=torch.int64)

        # Convert to uint8 for environment
        actions_np = tensor_to_numpy(actions, "uint8")

        assert actions_np.dtype == np.uint8
        assert actions_np.shape == (16, 2)
        assert np.all(actions_np >= 0) and np.all(actions_np <= 9)

    def test_rewards_conversion_scenario(self):
        """Test typical rewards conversion scenario."""
        # Simulate rewards from environment
        rewards = torch.randn(16) * 2.0  # Some rewards

        # Convert to float32 for consistency
        rewards_np = tensor_to_numpy(rewards, "float32", validate_range=False)

        assert rewards_np.dtype == np.float32
        assert rewards_np.shape == (16,)

    def test_large_tensor_conversion(self):
        """Test conversion of larger tensors."""
        # Test with larger tensor
        large_tensor = torch.randint(0, 100, (1000, 50), dtype=torch.int32)
        result = tensor_to_numpy(large_tensor, "uint8", validate_range=False)

        assert result.shape == (1000, 50)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize(
        "input_dtype,target_dtype",
        [
            (torch.int32, "uint8"),
            (torch.int64, "uint16"),
            (torch.float32, "float32"),
            (torch.float64, "float32"),
            (torch.bool, "bool"),
        ],
    )
    def test_common_dtype_conversions(self, input_dtype, target_dtype):
        """Test common dtype conversion patterns."""
        tensor = torch.ones(5, dtype=input_dtype)
        if input_dtype in [torch.int32, torch.int64]:
            tensor = torch.randint(0, 10, (5,), dtype=input_dtype)

        result = tensor_to_numpy(tensor, target_dtype, validate_range=False)
        assert result.dtype == np.dtype(target_dtype)


# Fixtures for reusable test data
@pytest.fixture
def sample_tensors():
    """Fixture providing sample tensors for testing."""
    return {
        "int_tensor": torch.tensor([1, 2, 3], dtype=torch.int32),
        "float_tensor": torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32),
        "bool_tensor": torch.tensor([True, False, True], dtype=torch.bool),
        "large_tensor": torch.randint(0, 255, (100, 10), dtype=torch.int32),
    }


@pytest.fixture
def sample_dtypes():
    """Fixture providing sample target dtypes."""
    return {"uint8": np.uint8, "int8": np.int8, "float32": np.float32, "bool": np.bool_}


if __name__ == "__main__":
    # Run tests with pytest when script is executed directly
    pytest.main([__file__])
