"""Directory management for Metta training runs."""

import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf

from metta.rl.trainer_config import TrainerConfig

logger = logging.getLogger(__name__)


class RunDirectories:
    """Container for run directory paths."""

    def __init__(self, run_dir: str, checkpoint_dir: str, replay_dir: str, stats_dir: str, run_name: str):
        self.run_dir = run_dir
        self.checkpoint_dir = checkpoint_dir
        self.replay_dir = replay_dir
        self.stats_dir = stats_dir
        self.run_name = run_name


def setup_run_directories(run_name: str | None = None, data_dir: str | None = None) -> RunDirectories:
    """Set up the directory structure for a training run.

    This creates the same directory structure as tools/train.py:
    - {data_dir}/{run_name}/
        - checkpoints/  # Model checkpoints
        - replays/      # Replay files
        - stats/        # Evaluation statistics

    Args:
        run_name: Name for this run. If not provided, uses METTA_RUN env var or timestamp
        data_dir: Base data directory. If not provided, uses DATA_DIR env var or ./train_dir

    Returns:
        RunDirectories object with all directory paths
    """
    # Get run name and data directory
    if run_name is None:
        run_name = os.environ.get("METTA_RUN", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if data_dir is None:
        data_dir = os.environ.get("DATA_DIR", "./train_dir")

    # Create paths
    run_dir = os.path.join(data_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    replay_dir = os.path.join(run_dir, "replays")
    stats_dir = os.path.join(run_dir, "stats")

    # Create directories
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(replay_dir).mkdir(parents=True, exist_ok=True)
    Path(stats_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Run name: {run_name}")

    return RunDirectories(
        run_dir=run_dir, checkpoint_dir=checkpoint_dir, replay_dir=replay_dir, stats_dir=stats_dir, run_name=run_name
    )


def save_experiment_config(
    dirs: RunDirectories,
    device: torch.device,
    trainer_config: TrainerConfig,
) -> None:
    """Save training configuration to config.yaml in the run directory.

    This builds the experiment configuration from the provided components
    and saves it for reproducibility. Only saves on master rank in distributed mode.

    Args:
        dirs: RunDirectories object with paths
        device: Training device
        trainer_config: TrainerConfig object with training parameters
    """
    # Only save on master rank to avoid file conflicts
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    # Build experiment configuration
    experiment_config = {
        "run": dirs.run_name,
        "run_dir": dirs.run_dir,
        "data_dir": os.path.dirname(dirs.run_dir),
        "device": str(device),
        "trainer": {
            "num_workers": trainer_config.num_workers,
            "total_timesteps": trainer_config.total_timesteps,
            "batch_size": trainer_config.batch_size,
            "minibatch_size": trainer_config.minibatch_size,
            "checkpoint_dir": dirs.checkpoint_dir,
            "optimizer": trainer_config.optimizer.model_dump(),
            "ppo": trainer_config.ppo.model_dump(),
            "checkpoint": trainer_config.checkpoint.model_dump(),
            "simulation": trainer_config.simulation.model_dump(),
            "profiler": trainer_config.profiler.model_dump(),
        },
    }

    # Save to file
    config_path = os.path.join(dirs.run_dir, "config.yaml")
    OmegaConf.save(experiment_config, config_path)
    logger.info(f"Saved config to {config_path}")


def setup_device_and_distributed(base_device: str = "cuda") -> tuple[torch.device, bool, int, int]:
    """Set up device and distributed training, returning all needed information.

    This combines device setup and distributed initialization into a single call,
    matching the initialization pattern from tools/train.py.

    Args:
        base_device: Base device string ("cuda" or "cpu")

    Returns:
        Tuple of (device, is_master, world_size, rank)
        - device: The torch.device to use for training
        - is_master: True if this is the master process (rank 0)
        - world_size: Total number of processes (1 if not distributed)
        - rank: Current process rank (0 if not distributed)
    """
    import torch

    # Check if we're in a distributed environment
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

        # For CUDA, use device with local rank
        if base_device.startswith("cuda"):
            device = torch.device(f"{base_device}:{local_rank}")
            backend = "nccl"
        else:
            # For CPU, just use cpu device (no rank suffix)
            device = torch.device(base_device)
            backend = "gloo"

        torch.distributed.init_process_group(backend=backend)

        # Get distributed info after initialization
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        is_master = rank == 0
    else:
        # Single GPU or CPU
        device = torch.device(base_device)
        rank = 0
        world_size = 1
        is_master = True

    logger.info(f"Using device: {device} (rank {rank}/{world_size})")
    return device, is_master, world_size, rank
