"""Runtime configuration for Metta - replaces common.yaml functionality."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch


@dataclass
class RuntimeConfig:
    """Runtime configuration that replaces common.yaml.

    This provides global settings that were previously scattered across
    various YAML files, now in a programmatic interface.
    """

    # Run identification
    run_name: str = "default_run"
    run_dir: Optional[Path] = None
    data_dir: Optional[Path] = None

    # Hardware settings
    device: Union[str, torch.device] = "cuda"
    torch_deterministic: bool = True
    seed: int = 0

    # Vectorization
    vectorization: str = "multiprocessing"  # or "serial", "ray"
    num_workers: int = 4

    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    # Logging
    log_level: str = "INFO"
    stats_user: Optional[str] = None

    def __post_init__(self):
        """Initialize derived values after dataclass creation."""
        # Set stats user from environment if not provided
        if self.stats_user is None:
            self.stats_user = os.environ.get("USER", "unknown")

        # Set data directory
        if self.data_dir is None:
            self.data_dir = Path(os.environ.get("DATA_DIR", "./train_dir"))

        # Set run directory
        if self.run_dir is None:
            self.run_dir = self.data_dir / self.run_name

        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        # Configure PyTorch
        if self.torch_deterministic:
            torch.use_deterministic_algorithms(True)

        # Set seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            import numpy as np

            np.random.seed(self.seed)
            import random

            random.seed(self.seed)

    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory."""
        return self.run_dir / "checkpoints"

    @property
    def policy_uri(self) -> str:
        """Get default policy URI."""
        return f"file://{self.checkpoint_dir}"

    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.distributed or self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0

    @classmethod
    def from_env(cls, **overrides) -> "RuntimeConfig":
        """Create runtime config from environment variables.

        This replaces the Hydra resolver functionality for environment variables.
        """
        config = cls(
            run_name=os.environ.get("RUN_NAME", overrides.get("run_name", "default_run")),
            device=os.environ.get("DEVICE", overrides.get("device", "cuda")),
            seed=int(os.environ.get("SEED", overrides.get("seed", 0))),
            **overrides,
        )

        # Check for distributed training environment variables
        if "RANK" in os.environ:
            config.distributed = True
            config.rank = int(os.environ["RANK"])
            config.world_size = int(os.environ.get("WORLD_SIZE", 1))
            config.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        return config

    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility with existing code."""
        return {
            "run": self.run_name,
            "run_dir": str(self.run_dir),
            "data_dir": str(self.data_dir),
            "device": str(self.device),
            "seed": self.seed,
            "vectorization": self.vectorization,
            "torch_deterministic": self.torch_deterministic,
            "stats_user": self.stats_user,
        }


# Global runtime instance (can be replaced/configured by user)
_runtime: Optional[RuntimeConfig] = None


def get_runtime() -> RuntimeConfig:
    """Get the global runtime configuration."""
    global _runtime
    if _runtime is None:
        _runtime = RuntimeConfig.from_env()
    return _runtime


def set_runtime(config: RuntimeConfig) -> None:
    """Set the global runtime configuration."""
    global _runtime
    _runtime = config


def configure(**kwargs) -> RuntimeConfig:
    """Configure runtime with keyword arguments."""
    config = RuntimeConfig(**kwargs)
    set_runtime(config)
    return config
