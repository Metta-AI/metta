"""
RNG State management for checkpoint system.

Provides Pydantic/OmegaConf-compatible configuration for saving and loading
random number generator states across all common RNG sources.
"""

import pickle
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from pydantic import Field

# Assuming your Config class is in a module called 'config'
# Adjust the import as needed for your project structure
from metta.common.util.config import Config  # Replace with your actual import path


class RngStateConfig(Config):
    """Configuration for random number generator states.

    Compatible with Pydantic BaseModel and OmegaConf. Captures states from:
    - PyTorch CPU random number generator
    - PyTorch CUDA random number generator(s)
    - NumPy random number generator
    - Python built-in random number generator
    """

    torch_cpu_state: Optional[bytes] = Field(default=None, description="PyTorch CPU RNG state")
    torch_cuda_states: Optional[List[bytes]] = Field(default=None, description="PyTorch CUDA RNG states")
    numpy_state: Optional[tuple] = Field(default=None, description="NumPy RNG state")
    python_state: Optional[tuple] = Field(default=None, description="Python RNG state")

    @classmethod
    def capture_current_states(cls) -> "RngStateConfig":
        """Capture the current state of all RNG sources.

        Returns:
            RngStateConfig with current RNG states populated
        """
        # Convert torch states to bytes for serialization
        torch_cpu_state = torch.get_rng_state().numpy().tobytes()

        torch_cuda_states = None
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # Get states from all CUDA devices and convert to bytes
            try:
                cuda_states = torch.cuda.get_rng_state_all()
                torch_cuda_states = [state.cpu().numpy().tobytes() for state in cuda_states]
            except RuntimeError:
                # Fallback for single device
                cuda_states = [torch.cuda.get_rng_state()]
                torch_cuda_states = [state.cpu().numpy().tobytes() for state in cuda_states]

        return cls(
            torch_cpu_state=torch_cpu_state,
            torch_cuda_states=torch_cuda_states,
            numpy_state=np.random.get_state(),
            python_state=random.getstate(),
        )

    def restore_states(self) -> None:
        """Restore all RNG states from this configuration."""
        # Force deterministic settings for maximum reproducibility
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # CPU-specific deterministic settings
        torch.set_num_threads(1)  # Force single-threaded CPU operations
        torch.set_num_interop_threads(1)  # Disable inter-op parallelism

        # Set environment variables for BLAS libraries
        import os

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        # Seems redundant
        torch.manual_seed(1337)
        np.random.seed(1337)
        random.seed(1337)

        if self.torch_cpu_state is not None:
            # Convert bytes back to tensor
            state_array = np.frombuffer(self.torch_cpu_state, dtype=np.uint8)
            torch_state = torch.from_numpy(state_array)
            torch.set_rng_state(torch_state)

        if self.torch_cuda_states is not None:
            # Convert bytes back to tensors
            cuda_states = []
            for state_bytes in self.torch_cuda_states:
                state_array = np.frombuffer(state_bytes, dtype=np.uint8)
                cuda_states.append(torch.from_numpy(state_array))
            torch.cuda.set_rng_state_all(cuda_states)

        if self.numpy_state is not None:
            np.random.set_state(self.numpy_state)

        if self.python_state is not None:
            random.setstate(self.python_state)

    def save(self, path: str) -> None:
        """Save RNG states to file.

        Args:
            path: File path to save to (will create parent directories if needed)
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Use model_dump() to get serializable dictionary
        state_dict = self.model_dump()

        with open(path_obj, "wb") as f:
            pickle.dump(state_dict, f)

    @classmethod
    def load(cls, path: str) -> "RngStateConfig":
        """Load RNG states from file.

        Args:
            path: File path to load from

        Returns:
            RngStateConfig with loaded states

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"RNG state file not found: {path}")

        with open(path_obj, "rb") as f:
            state_dict = pickle.load(f)

        # Create instance using the loaded dictionary
        return cls(**state_dict)


def save_current_rng_state(path: str) -> None:
    """Convenience function to capture and save current RNG state.

    Args:
        path: File path to save RNG state to
    """
    rng_config = RngStateConfig.capture_current_states()
    rng_config.save(path)


def load_and_restore_rng_state(path: str) -> RngStateConfig:
    """Convenience function to load and restore RNG state.

    Args:
        path: File path to load RNG state from

    Returns:
        The loaded RngStateConfig (in case you need it for other purposes)
    """
    # Force deterministic settings for maximum reproducibility
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rng_config = RngStateConfig.load(path)
    rng_config.restore_states()
    return rng_config


def save_rng_state_to_yaml(path: str) -> None:
    """Save current RNG state as YAML (for debugging/inspection).

    Note: This saves a YAML representation but the actual tensor data
    is still binary. Use save_current_rng_state() for full functionality.

    Args:
        path: File path to save YAML to
    """
    rng_config = RngStateConfig.capture_current_states()
    yaml_str = rng_config.yaml()

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w") as f:
        f.write(yaml_str)
