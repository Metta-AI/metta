"""Policy management utilities for Metta."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent, PolicyAgent
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.checkpoint_manager import parse_checkpoint_filename
from metta.rl.wandb import get_wandb_artifact_metadata, load_policy_from_wandb_uri

logger = logging.getLogger(__name__)


def validate_policy_environment_match(policy: PolicyAgent, env: MettaGridEnv) -> None:
    """Validate that policy's observation shape matches environment's."""
    # Extract agent from distributed wrapper if needed
    if isinstance(policy, MettaAgent):
        agent = policy
    elif isinstance(policy, DistributedMettaAgent):
        agent = policy.module

    elif type(policy).__name__ == "Recurrent":
        agent = policy
    else:
        raise ValueError(f"Policy must be of type MettaAgent or DistributedMettaAgent, got {type(policy)}")

    _env_shape = env.single_observation_space.shape
    environment_shape = tuple(_env_shape) if isinstance(_env_shape, list) else _env_shape

    # The rest of the validation logic continues to work with duck typing
    if hasattr(agent, "components"):
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
                "No component with observation shape found in policy. "
                f"Environment observation shape: {environment_shape}"
            )


def wrap_agent_distributed(agent: PolicyAgent, device: torch.device) -> PolicyAgent:
    if torch.distributed.is_initialized():
        # Always use DistributedMettaAgent for its __getattr__ forwarding
        agent = DistributedMettaAgent(agent, device)

    return agent


# URI Resolution Functions (moved from policy_uri_resolver.py)


def _checkpoint_tuple_to_dict(filename: str) -> Dict[str, Any]:
    """Convert checkpoint filename to metadata dict for compatibility."""
    try:
        run_name, epoch, agent_step, total_time = parse_checkpoint_filename(filename)
        return {
            "run": run_name,
            "epoch": epoch,
            "agent_step": agent_step,
            "total_time": total_time,
            "checkpoint_file": filename,
        }
    except ValueError:
        return {}


def resolve_policy(uri: str, device: str = "cpu") -> torch.nn.Module:
    """Load a policy from file:// or wandb:// URI."""
    if uri.startswith("file://"):
        file_path = Path(uri[7:])
        if file_path.is_file():
            return torch.load(file_path, map_location=device, weights_only=False)
        elif file_path.is_dir():
            # Find latest checkpoint in directory
            pattern = "*.e*.s*.t*.pt"
            checkpoints = list(file_path.glob(pattern))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint files found in {file_path}")
            latest = max(checkpoints, key=lambda f: parse_checkpoint_filename(f.name)[1])
            return torch.load(latest, map_location=device, weights_only=False)
        else:
            raise FileNotFoundError(f"Path does not exist: {file_path}")
    elif uri.startswith("wandb://"):
        return load_policy_from_wandb_uri(uri, device)
    else:
        raise ValueError(f"Unsupported URI scheme: {uri}")


def get_policy_metadata(uri: str) -> Dict[str, Any]:
    """Get metadata from a policy URI."""
    if uri.startswith("file://"):
        file_path = Path(uri[7:])
        if file_path.is_file():
            return _checkpoint_tuple_to_dict(file_path.name)
        elif file_path.is_dir():
            # Find latest checkpoint in directory
            pattern = "*.e*.s*.t*.pt"
            checkpoints = list(file_path.glob(pattern))
            if not checkpoints:
                return {}
            latest = max(checkpoints, key=lambda f: parse_checkpoint_filename(f.name)[1])
            return _checkpoint_tuple_to_dict(latest.name)
        else:
            return {}
    elif uri.startswith("wandb://"):
        return get_wandb_artifact_metadata(uri)
    else:
        raise ValueError(f"Unsupported URI scheme: {uri}")


def discover_policies(
    base_uri: str, strategy: str = "latest", count: int = 1, metric: str = "epoch"
) -> List[Tuple[str, Dict[str, Any]]]:
    """Discover policies from a base URI."""
    if base_uri.startswith("file://"):
        dir_path = Path(base_uri[7:])
        pattern = "*.e*.s*.t*.pt"
        checkpoints = list(dir_path.glob(pattern))
        if not checkpoints:
            return []

        # Sort checkpoints based on strategy
        if strategy == "latest":
            checkpoints.sort(key=lambda f: parse_checkpoint_filename(f.name)[1], reverse=True)
        elif strategy in ["best_score", "top"]:
            metric_idx = {"epoch": 1, "agent_step": 2, "total_time": 3}.get(metric, 1)
            checkpoints.sort(key=lambda f: parse_checkpoint_filename(f.name)[metric_idx], reverse=True)
        elif strategy != "all":
            raise ValueError(f"Unknown strategy: {strategy}")

        # Return requested count
        if strategy == "all":
            count = len(checkpoints)

        results = []
        for checkpoint in checkpoints[:count]:
            uri = f"file://{checkpoint}"
            metadata = _checkpoint_tuple_to_dict(checkpoint.name)
            results.append((uri, metadata))

        return results
    elif base_uri.startswith("wandb://"):
        metadata = get_policy_metadata(base_uri)
        return [(base_uri, metadata)]
    else:
        raise ValueError(f"Unsupported URI scheme: {base_uri}")


# Convenience aliases
resolve_policy_from_uri = resolve_policy
get_policy_metadata_from_uri = get_policy_metadata
discover_policies_from_uri = discover_policies
