import logging
from pathlib import Path
from typing import List

import torch

from metta.agent.metta_agent import DistributedMettaAgent, PolicyAgent
from metta.rl.checkpoint_manager import CheckpointManager, get_checkpoint_uri_from_dir
from metta.rl.wandb import load_policy_from_wandb_uri

logger = logging.getLogger(__name__)


def wrap_agent_distributed(agent: PolicyAgent, device: torch.device) -> PolicyAgent:
    if torch.distributed.is_initialized():
        return DistributedMettaAgent(agent, device)
    return agent


def resolve_policy(uri: str, device: str = "cpu") -> torch.nn.Module:
    """Load a policy from file:// or wandb:// URI. Simplified to handle common cases."""
    if uri.startswith("file://"):
        file_path = Path(uri[7:])

        # If it's a directory, find the latest checkpoint
        if file_path.is_dir():
            uri = get_checkpoint_uri_from_dir(str(file_path))
            file_path = Path(uri[7:])

        # Load the checkpoint file directly - let torch.load raise if not found
        return torch.load(file_path, map_location=device, weights_only=False)

    elif uri.startswith("wandb://"):
        return load_policy_from_wandb_uri(uri, device)
    else:
        raise ValueError(f"Unsupported URI scheme: {uri}")


def discover_policy_uris(base_uri: str, strategy: str = "latest", count: int = 1, metric: str = "epoch") -> List[str]:
    """Discover policy URIs from a base URI using CheckpointManager."""
    if base_uri.startswith("file://"):
        dir_path = Path(base_uri[7:])

        # Determine run_dir and run_name
        if dir_path.name == "checkpoints":
            run_name = dir_path.parent.name
            run_dir = str(dir_path.parent.parent)
        else:
            run_name = dir_path.name
            run_dir = str(dir_path.parent)

        # Use CheckpointManager to select checkpoints
        checkpoint_manager = CheckpointManager(run_name=run_name, run_dir=run_dir)
        checkpoint_paths = checkpoint_manager.select_checkpoints(strategy=strategy, count=count, metric=metric)

        # Convert paths to URIs
        return [f"file://{path}" for path in checkpoint_paths]
    elif base_uri.startswith("wandb://"):
        return [base_uri]
    else:
        raise ValueError(f"Unsupported URI scheme: {base_uri}")
