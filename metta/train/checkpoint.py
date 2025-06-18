"""Checkpoint management for training."""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    checkpoint_dir: str,
    agent,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    agent_step: int,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save training checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoints
        agent: The policy/agent to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        agent_step: Current agent step count
        lr_scheduler: Optional LR scheduler to save
        metadata: Optional metadata to include

    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create checkpoint dict
    checkpoint = {
        "epoch": epoch,
        "agent_step": agent_step,
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    if lr_scheduler is not None:
        checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

    if metadata is not None:
        checkpoint["metadata"] = metadata

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch:06d}.pt")
    torch.save(checkpoint, checkpoint_path)

    # Update latest symlink
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.basename(checkpoint_path), latest_path)

    # Save metadata separately for easy access
    metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "latest_epoch": epoch,
                "latest_agent_step": agent_step,
                "latest_checkpoint": checkpoint_path,
                "timestamp": checkpoint["timestamp"],
            },
            f,
            indent=2,
        )

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    agent,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        agent: The policy/agent to load weights into
        optimizer: Optional optimizer to load state into
        lr_scheduler: Optional LR scheduler to load state into
        device: Device to load tensors to

    Returns:
        Dictionary with checkpoint information
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load agent weights
    agent.load_state_dict(checkpoint["agent_state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load LR scheduler state if provided
    if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "agent_step": checkpoint.get("agent_step", 0),
        "timestamp": checkpoint.get("timestamp"),
        "metadata": checkpoint.get("metadata", {}),
    }


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        Path to latest checkpoint or None if not found
    """
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest_path):
        # Follow symlink
        return os.path.join(checkpoint_dir, os.readlink(latest_path))

    # Fall back to searching for checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]

    if not checkpoint_files:
        return None

    # Sort by epoch number
    checkpoint_files.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
    return os.path.join(checkpoint_dir, checkpoint_files[-1])
