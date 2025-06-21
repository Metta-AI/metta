from typing import Optional, Tuple, Union

import torch


def assert_shape(tensor: torch.Tensor, expected_shape: Tuple[Union[int, str], ...], name: Optional[str] = None) -> bool:
    """
    Assert that a tensor has an expected shape by raising ValueError if the shape doesn't match. This function should
    be called from within a debug block to allow optimization to skip the checks

    Args:
        tensor: The tensor to check
        expected_shape: A tuple specifying expected dimensions. Each element can be:
            - An integer: Requires the dimension to have exactly this size
            - A string: Represents a named dimension (like "B", "T", "H", "W") where any
              positive size is acceptable
        name: Optional name for the tensor to include in the error message

    Examples:
        if __debug__:
            assert_shape(logits, ("B", "T", 10000), 'logits')

    Raises:
        ValueError: If the tensor shape doesn't match the expected shape

    Returns:
        bool: True when the assertion passes
    """
    tensor_shape = tuple(tensor.shape)
    tensor_name = f"'{name}'" if name else "tensor"

    # Check number of dimensions
    if len(tensor_shape) != len(expected_shape):
        raise ValueError(
            f"{tensor_name} has shape {tensor_shape} with {len(tensor_shape)} dimensions."
            f" We expected {len(expected_shape)} dimensions."
        )

    # Check each dimension
    for i, (actual, expected) in enumerate(zip(tensor_shape, expected_shape, strict=False)):
        if isinstance(expected, int) and actual != expected:
            raise ValueError(f"{tensor_name} dimension {i} has size {actual}, expected {expected}")
        elif isinstance(expected, str) and actual <= 0:
            raise ValueError(
                f"{tensor_name} dimension {i} ('{expected}') has invalid size {actual}, expected a positive value"
            )

    return True  # Return True when the assertion passes
