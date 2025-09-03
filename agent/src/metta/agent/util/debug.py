from typing import Optional, Tuple, Union

import torch


def assert_shape(tensor: torch.Tensor, expected_shape: Tuple[Union[int, str], ...], name: Optional[str] = None) -> bool:
    """Assert tensor has expected shape, allowing named dimensions (strings) for variable sizes."""
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
