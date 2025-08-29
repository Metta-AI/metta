"""Policy management utilities for Metta."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from metta.agent.metta_agent import DistributedMettaAgent, PolicyAgent
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.checkpoint_manager import (
    CheckpointManager,
    get_checkpoint_from_dir,
    get_checkpoint_uri_from_dir,
    metadata_from_uri,
    parse_checkpoint_filename,
)
from metta.rl.wandb import get_wandb_artifact_metadata, load_policy_from_wandb_uri

logger = logging.getLogger(__name__)


def validate_policy_environment_match(policy: PolicyAgent, env: MettaGridEnv) -> None:
    """Validate that policy's observation shape matches environment's."""
    # Extract agent from distributed wrapper if needed
    agent = policy.module if isinstance(policy, DistributedMettaAgent) else policy

    _env_shape = env.single_observation_space.shape
    environment_shape = tuple(_env_shape) if isinstance(_env_shape, list) else _env_shape

    # Validate observation shapes match
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


# URI Resolution Functions (moved from policy_uri_resolver.py)


def resolve_policy(uri: str, device: str = "cpu") -> torch.nn.Module:
    """Load a policy from file:// or wandb:// URI."""
    if uri.startswith("file://"):
        file_path = Path(uri[7:])
        if file_path.is_file():
            return torch.load(file_path, map_location=device, weights_only=False)
        elif file_path.is_dir():
            result = get_checkpoint_from_dir(str(file_path))
            if result:
                uri, policy = result
                return policy
            raise FileNotFoundError(f"No checkpoint found in directory: {file_path}")
        else:
            raise FileNotFoundError(f"Path does not exist: {file_path}")
    elif uri.startswith("wandb://"):
        return load_policy_from_wandb_uri(uri, device)
    else:
        raise ValueError(f"Unsupported URI scheme: {uri}")


def get_policy_metadata(uri: str) -> Dict[str, Any]:
    """Get metadata from a policy URI - simplified version."""
    if uri.startswith("file://"):
        file_path = Path(uri[7:])
        if file_path.is_dir():
            # Convert directory to actual file URI
            uri = get_checkpoint_uri_from_dir(str(file_path))
            if not uri:
                return {}
        return metadata_from_uri(uri)
    elif uri.startswith("wandb://"):
        return get_wandb_artifact_metadata(uri)
    else:
        return metadata_from_uri(uri)


def discover_policies(
    base_uri: str, strategy: str = "latest", count: int = 1, metric: str = "epoch"
) -> List[Tuple[str, Dict[str, Any]]]:
    """Discover policies from a base URI."""
    if base_uri.startswith("file://"):
        dir_path = Path(base_uri[7:])
        run_name = dir_path.name
        run_dir = dir_path.parent
        manager = CheckpointManager(run_name, str(run_dir))

        # Use CheckpointManager's select_checkpoints method
        checkpoints = manager.select_checkpoints(strategy, count, metric)

        results = []
        for checkpoint in checkpoints:
            uri = f"file://{checkpoint}"
            run_name, epoch, agent_step, total_time = parse_checkpoint_filename(checkpoint.name)
            metadata = {
                "run": run_name,
                "epoch": epoch,
                "agent_step": agent_step,
                "total_time": total_time,
                "checkpoint_file": checkpoint.name,
            }
            results.append((uri, metadata))

        return results
    elif base_uri.startswith("wandb://"):
        metadata = get_policy_metadata(base_uri)
        return [(base_uri, metadata)]
    else:
        raise ValueError(f"Unsupported URI scheme: {base_uri}")
