from typing import Union

import numpy as np
import torch


def tensor_to_numpy(
    tensor: torch.Tensor, target_dtype: Union[np.dtype, type, str], validate_range: bool = __debug__, copy: bool = True
) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array with safe type conversion.

    Args:
        tensor: PyTorch tensor to convert
        target_dtype: Target NumPy dtype (can be np.dtype, type like np.uint8, or string like 'uint8')
        validate_range: Whether to validate values are within target type range.
                       Defaults to __debug__ so validation is compiled out in optimized builds.
        copy: Whether to force a copy. Defaults to True for safety.

    Returns:
        NumPy array with correct dtype

    Raises:
        AssertionError: If tensor values are outside valid range for target dtype (debug builds only)
        TypeError: If tensor is not a PyTorch tensor
        ValueError: If target_dtype is invalid
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    # Normalize dtype
    target_dtype = np.dtype(target_dtype)

    # Convert to numpy efficiently
    if tensor.is_cuda:
        numpy_array = tensor.detach().cpu().numpy()  # detach() for safety
    else:
        # Use .detach() to avoid potential autograd issues
        numpy_array = tensor.detach().numpy()

    # Early return if no conversion needed and no copy requested
    if numpy_array.dtype == target_dtype and copy is False:
        return numpy_array

    if validate_range:
        _validate_array_range(numpy_array, target_dtype)

    # Handle copy parameter
    if copy is None:
        copy = numpy_array.dtype != target_dtype

    return numpy_array.astype(target_dtype, copy=copy)


def _validate_array_range(array: np.ndarray, target_dtype: np.dtype) -> None:
    """Internal function to validate array range fits in target dtype."""
    if np.issubdtype(target_dtype, np.integer):
        type_info = np.iinfo(target_dtype)
        min_val, max_val = type_info.min, type_info.max

        # Use numpy functions for efficiency
        array_min, array_max = np.min(array), np.max(array)
        assert array_min >= min_val and array_max <= max_val, (
            f"Array values must be in range [{min_val}, {max_val}] for {target_dtype}, "
            f"got range [{array_min}, {array_max}]"
        )

    elif np.issubdtype(target_dtype, np.floating):
        type_info = np.finfo(target_dtype)
        # For floats, we mainly care about finite values
        if not np.isfinite(array).all():
            assert target_dtype in [np.float32, np.float64], f"Non-finite values not supported for {target_dtype}"

        # Check for overflow to infinity
        array_min, array_max = np.min(array), np.max(array)
        if np.isfinite(array_min) and np.isfinite(array_max):
            # Only check range for finite values
            assert array_min >= type_info.min and array_max <= type_info.max, (
                f"Array values must be in range [{type_info.min}, {type_info.max}] for {target_dtype}, "
                f"got range [{array_min}, {array_max}]"
            )


def validate_tensor_range(
    tensor: torch.Tensor,
    min_val: Union[float, int, np.floating, np.integer],
    max_val: Union[float, int, np.floating, np.integer],
) -> None:
    """
    Validate that tensor values are within specified range (debug builds only).

    Args:
        tensor: PyTorch tensor to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Raises:
        AssertionError: If values are outside range (debug builds only)
    """
    if __debug__:
        # Use torch operations for efficiency
        tensor_min, tensor_max = torch.min(tensor).item(), torch.max(tensor).item()
        assert tensor_min >= min_val and tensor_max <= max_val, (
            f"Tensor values must be in range [{min_val}, {max_val}], got range [{tensor_min}, {tensor_max}]"
        )


def validate_tensor_dtype_range(tensor: torch.Tensor, target_dtype: Union[np.dtype, type, str]) -> None:
    """
    Validate that tensor values fit within target numpy dtype range (debug builds only).

    Args:
        tensor: PyTorch tensor to validate
        target_dtype: Target NumPy dtype to check against

    Raises:
        AssertionError: If values are outside dtype range (debug builds only)
    """
    if __debug__:
        target_dtype = np.dtype(target_dtype)

        if np.issubdtype(target_dtype, np.integer):
            type_info = np.iinfo(target_dtype)
            validate_tensor_range(tensor, float(type_info.min), float(type_info.max))
        elif np.issubdtype(target_dtype, np.floating):
            type_info = np.finfo(target_dtype)
            validate_tensor_range(tensor, float(type_info.min), float(type_info.max))


def batch_tensor_to_numpy(
    tensors: dict[str, torch.Tensor],
    target_dtypes: dict[str, Union[np.dtype, type, str]],
    validate_range: bool = __debug__,
) -> dict[str, np.ndarray]:
    """
    Convert multiple tensors to numpy arrays efficiently.

    Args:
        tensors: Dictionary of name -> tensor
        target_dtypes: Dictionary of name -> target dtype
        validate_range: Whether to validate ranges

    Returns:
        Dictionary of name -> numpy array
    """
    result = {}
    for name, tensor in tensors.items():
        if name in target_dtypes:
            result[name] = tensor_to_numpy(tensor, target_dtypes[name], validate_range)
        else:
            # Convert without type change
            result[name] = tensor_to_numpy(tensor, tensor.numpy().dtype, validate_range=False)
    return result


def get_dtype_info(dtype: Union[np.dtype, type, str]) -> dict:
    """
    Get comprehensive information about a numpy dtype.

    Args:
        dtype: NumPy dtype to inspect

    Returns:
        Dictionary with dtype information
    """
    dtype = np.dtype(dtype)
    info = {
        "dtype": dtype,
        "name": dtype.name,
        "kind": dtype.kind,
        "itemsize": dtype.itemsize,
        "is_integer": np.issubdtype(dtype, np.integer),
        "is_floating": np.issubdtype(dtype, np.floating),
        "is_signed": dtype.kind in "if",
    }

    if info["is_integer"]:
        type_info = np.iinfo(dtype)
        info.update(
            {
                "min": type_info.min,
                "max": type_info.max,
                "bits": type_info.bits,
            }
        )
    elif info["is_floating"]:
        type_info = np.finfo(dtype)
        info.update(
            {
                "min": type_info.min,
                "max": type_info.max,
                "eps": type_info.eps,
                "precision": type_info.precision,
            }
        )

    return info


# Example usage and testing
if __name__ == "__main__":
    print("Testing improved tensor_to_numpy conversions...")

    # Test basic conversion
    tensor = torch.randint(0, 10, (4, 2), dtype=torch.int32)
    print(f"Original: {tensor.dtype}, shape: {tensor.shape}")

    # Test different dtype input formats
    result1 = tensor_to_numpy(tensor, np.uint8)
    result2 = tensor_to_numpy(tensor, "uint8")
    result3 = tensor_to_numpy(tensor, np.dtype("uint8"))

    print(f"All conversions equal: {np.array_equal(result1, result2) and np.array_equal(result2, result3)}")

    # Test batch conversion
    tensors = {
        "actions": torch.randint(0, 5, (2, 2)),
        "rewards": torch.randn(2),
        "done": torch.tensor([0, 1], dtype=torch.bool),
    }

    dtypes = {"actions": "uint8", "rewards": "float32", "done": "bool"}

    results = batch_tensor_to_numpy(tensors, dtypes)
    print(f"Batch results: {[(k, v.dtype) for k, v in results.items()]}")

    # Test dtype info
    info = get_dtype_info("uint8")
    print(f"uint8 info: {info}")

    print("All tests passed!")
