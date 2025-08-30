import logging
from pathlib import Path
from typing import List

import torch

from metta.agent.metta_agent import DistributedMettaAgent, PolicyAgent
from metta.rl.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


def wrap_agent_distributed(agent: PolicyAgent, device: torch.device) -> PolicyAgent:
    if torch.distributed.is_initialized():
        return DistributedMettaAgent(agent, device)
    return agent


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
    elif base_uri.startswith("mock://"):
        # Handle mock URIs for testing - return the URI as-is
        return [base_uri]
    else:
        raise ValueError(f"Unsupported URI scheme: {base_uri}")
