"""Policy management utilities for Metta."""

import logging
from pathlib import Path
from typing import List

import torch

from metta.agent.metta_agent import DistributedMettaAgent, PolicyAgent
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.checkpoint_manager import get_checkpoint_uri_from_dir
from metta.rl.wandb import load_policy_from_wandb_uri

logger = logging.getLogger(__name__)


def validate_policy_environment_match(policy: PolicyAgent, env: MettaGridEnv) -> None:
    """Validate that policy's observation shape matches environment's."""
    agent = policy.module if isinstance(policy, DistributedMettaAgent) else policy

    _env_shape = env.single_observation_space.shape
    environment_shape = tuple(_env_shape) if isinstance(_env_shape, list) else _env_shape

    found_match = False
    for component_name, component in agent.components.items():
        if hasattr(component, "_obs_shape"):
            found_match = True
            component_shape = (
                tuple(component._obs_shape) if isinstance(component._obs_shape, list) else component._obs_shape
            )
            if component_shape != environment_shape:
                raise ValueError(
                    f"Observation space mismatch error:\n"
                    f"[policy] component_name: {component_name}\n"
                    f"[policy] component_shape: {component_shape}\n"
                    f"environment_shape: {environment_shape}\n"
                )

    if not found_match:
        raise ValueError(
            f"No component with observation shape found in policy. Environment observation shape: {environment_shape}"
        )


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
            if not uri:
                raise FileNotFoundError(f"No checkpoint found in directory: {file_path}")
            file_path = Path(uri[7:])
        
        # Load the checkpoint file
        if not file_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {file_path}")
        return torch.load(file_path, map_location=device, weights_only=False)
        
    elif uri.startswith("wandb://"):
        return load_policy_from_wandb_uri(uri, device)
    else:
        raise ValueError(f"Unsupported URI scheme: {uri}")


def discover_policy_uris(base_uri: str, strategy: str = "latest", count: int = 1, metric: str = "epoch") -> List[str]:
    """Discover policy URIs from a base URI. Simplified to work directly with files."""
    if base_uri.startswith("file://"):
        dir_path = Path(base_uri[7:])
        
        # Look for checkpoint files directly - no need for CheckpointManager
        checkpoint_dir = dir_path / "checkpoints" if (dir_path / "checkpoints").exists() else dir_path
        run_name = dir_path.name
        
        # Find all checkpoint files matching the pattern
        checkpoint_files = list(checkpoint_dir.glob(f"{run_name}.e*.s*.t*.pt"))
        if not checkpoint_files:
            return []
        
        # Parse and sort by the selected metric
        from metta.rl.checkpoint_manager import parse_checkpoint_filename
        metric_idx = {"epoch": 1, "agent_step": 2, "total_time": 3}.get(metric, 1)
        checkpoint_files.sort(key=lambda f: parse_checkpoint_filename(f.name)[metric_idx], reverse=True)
        
        # Return requested number of URIs
        selected = checkpoint_files if strategy == "all" else checkpoint_files[:count]
        return [f"file://{checkpoint}" for checkpoint in selected]
    elif base_uri.startswith("wandb://"):
        return [base_uri]
    else:
        raise ValueError(f"Unsupported URI scheme: {base_uri}")
