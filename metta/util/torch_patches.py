import torch


def safe_torch_load(path, map_location=None, weights_only=False):
    """
    A helper function that safely loads PyTorch models, handling MPS device issues.

    This function addresses known issues with PyTorch's MPS backend implementation:
    - GitHub Issue #127676: "module 'torch.mps' has no attribute 'device'"
      https://github.com/pytorch/pytorch/issues/127676
    - Similar issue with missing 'current_device' method in the torch.mps module

    This is a temporary workaround until these issues are fixed in PyTorch upstream.
    The patch adds missing methods to the torch.mps module to prevent errors when
    loading models that were trained on Apple Silicon GPUs.

    Args:
        path: Path to the PyTorch model file
        map_location: Device to map model to (default: None)
        weights_only: Whether to load only weights (default: False)

    Returns:
        The loaded PyTorch model/object
    """
    # type: ignore[attr-defined] # These attributes don't exist - we're adding them

    # Add MPS compatibility patch for 'current_device' method
    # Referenced in PyTorch issue #127676 and related issues
    if hasattr(torch, "mps") and not hasattr(torch.mps, "current_device"):
        # This mock implementation returns 0 as there's only one MPS device
        torch.mps.current_device = lambda: 0  # type: ignore

    # Add MPS compatibility patch for 'device' context manager
    # Referenced in PyTorch issue #127676
    if hasattr(torch, "mps") and not hasattr(torch.mps, "device"):
        # This provides a dummy context manager that does nothing
        class DummyDeviceContext:
            def __init__(self, idx):
                pass

            def __enter__(self):
                pass

            def __exit__(self, *args):
                pass

        torch.mps.device = DummyDeviceContext  # type: ignore

    # If explicitly requested to use MPS but we know it might fail,
    # redirect to CPU first to avoid potential issues
    if isinstance(map_location, str) and "mps" in map_location:
        loaded = safe_torch_load(path, map_location="cpu", weights_only=weights_only)
        # The model will still be on CPU, but can be moved to MPS later if needed
        return loaded

    # Otherwise, use the provided map_location
    returnsafe_torch_load(path, map_location=map_location, weights_only=weights_only)
