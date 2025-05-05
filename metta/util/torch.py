import logging
import os
from typing import Optional, Tuple

import torch


def check_policy_compatibility(path: str, warn_only: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Checks if a PyTorch policy file is compatible with the current hardware.

    This function examines a saved PyTorch model to determine if it was trained on
    hardware (like CUDA or MPS) that isn't available on the current machine.

    Args:
        path: Path to the PyTorch model file
        warn_only: If True, only log warnings instead of raising exceptions

    Returns:
        Tuple of (is_compatible, device_type):
            - is_compatible: Boolean indicating if the model is compatible
            - device_type: String indicating the device the model was saved on (or None if unknown)

    Raises:
        RuntimeError: If the model is incompatible and warn_only=False
    """
    # Check if the file exists
    if not os.path.exists(path):
        msg = f"Model file not found: {path}"
        if warn_only:
            logging.error(msg)
            return False, None
        else:
            raise FileNotFoundError(msg)

    # Patch MPS-related methods for Apple Silicon compatibility
    if hasattr(torch, "mps") and not hasattr(torch.mps, "current_device"):
        torch.mps.current_device = lambda: 0  # type: ignore

    if hasattr(torch, "mps") and not hasattr(torch.mps, "device"):

        class DummyDeviceContext:
            def __init__(self, idx):
                pass

            def __enter__(self):
                pass

            def __exit__(self, *args):
                pass

        torch.mps.device = DummyDeviceContext  # type: ignore

    try:
        # Try to peek at the model's metadata without fully loading it
        # This approach doesn't fully load the model but gets device info
        device_type = None
        with open(path, "rb") as f:
            try:
                # Try to read the first part of the file to detect device info
                import pickle

                unpickler = pickle.Unpickler(f)

                # Skip ahead to try to find device info
                # This is a heuristic approach and might not work for all models
                for _ in range(10):  # Try the first few objects
                    try:
                        obj = unpickler.load()
                        if isinstance(obj, dict) and "_metadata" in obj:
                            break
                    except EOFError:
                        break
                    except Exception:
                        continue
            except Exception:
                # If any error occurs during the peek, we'll do a full load
                pass

        # If metadata approach didn't work, try to actually load the model to CPU
        # to check its parameters' device information
        try:
            model = torch.load(path, map_location="cpu")

            # If model has parameters, check their original device
            if hasattr(model, "parameters"):
                for param in model.parameters():
                    if param.device.type == "cuda":
                        device_type = "cuda"
                        break
                    elif param.device.type == "mps":
                        device_type = "mps"
                        break
        except Exception as e:
            logging.warning(f"Couldn't inspect model parameters: {e}")

        # If we found a device type, check compatibility
        if device_type == "cuda" and not torch.cuda.is_available():
            msg = (
                f"The model at {path} was trained on a CUDA GPU, but CUDA is not available "
                "on this machine. The model can still be loaded, but you might experience "
                "compatibility issues. Consider using CPU mode."
            )
            if warn_only:
                logging.warning(msg)
                return False, device_type
            else:
                raise RuntimeError(msg)

        elif device_type == "mps" and not hasattr(torch, "mps"):
            msg = (
                f"The model at {path} was trained on an Apple MPS device, but MPS is not "
                "available on this machine. The model can still be loaded, but you might "
                "experience compatibility issues. Consider using CPU mode."
            )
            if warn_only:
                logging.warning(msg)
                return False, device_type
            else:
                raise RuntimeError(msg)

        # If we reach here, the model is likely compatible
        return True, device_type

    except (RuntimeError, FileNotFoundError):
        # Pass through our own exceptions
        if warn_only:
            return False, None
        else:
            raise
    except Exception as e:
        # Catch any other exceptions that might occur
        msg = f"Error checking model compatibility: {e}"
        if warn_only:
            logging.error(msg)
            return False, None
        else:
            raise RuntimeError(msg) from e
