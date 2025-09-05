"""Policy artifact utilities for Metta RL framework.

This module provides functions for loading, saving, and converting policy artifacts
between different formats (PyTorch .pt files and safetensors format).
"""

import json
from pathlib import Path
from typing import Any, Dict

import torch

from metta.agent.metta_agent import MettaAgent, PolicyAgent
from metta.agent.agent_config import AgentConfig
from metta.agent.agent_env_config import AgentEnvConfig


def write_agent_to_file(agent: Any, write_to: str) -> None:
    """Write an agent to a .pt file via pickling.

    Args:
        agent: The agent object to save
        write_to: Base path for the output file (without extension)
    """
    base_path = Path(write_to)
    pt_path = base_path.with_suffix(".pt")
    torch.save(agent, pt_path)


def from_weights(weights: Dict[str, torch.Tensor], init_config: AgentEnvConfig) -> PolicyAgent:
    """Create a PolicyAgent from weights.

    Args:
        weights: Dictionary mapping parameter names to tensors
        init_config: Agent environment configuration

    Returns:
        PolicyAgent with weights loaded
    """
    return MettaAgent.from_weights(weights, init_config, AgentConfig.model_validate(init_config.agent_config))


def save_safetensors(weights: Dict[str, torch.Tensor], statistics: Dict[str, Any],
                    init_config: AgentEnvConfig, write_to: str) -> None:
    """Save weights and statistics to safetensors format.

    Args:
        weights: Dictionary mapping parameter names to tensors
        statistics: Policy statistics
        init_config: Agent environment configuration
        write_to: Base path for output files (without extension)
    """
    base_path = Path(write_to)

    # Write weights to .safetensors
    try:
        from safetensors.torch import save_file
        safetensors_path = base_path.with_suffix(".safetensors")
        save_file(weights, str(safetensors_path))
    except ImportError:
        # Fallback to torch.save if safetensors not available
        pt_path = base_path.with_suffix(".pt")
        torch.save(weights, pt_path)

    # Write statistics to .stats file
    stats_path = base_path.with_suffix(".stats")
    with open(stats_path, "w") as f:
        json.dump(statistics, f, indent=2)

    # Write init config to .init_config file
    if init_config is not None:
        init_config_path = base_path.with_suffix(".init_config")
        with open(init_config_path, "w") as f:
            f.write(init_config.model_dump_json(indent=2))


def get_statistics_from_path(base_path: str | Path) -> Dict[str, Any]:
    """Get statistics from a checkpoint base path without loading the full artifact.

    This function is useful when you only need the statistics/metadata and don't want
    to load the potentially large weights into memory.

    Args:
        base_path: Base path for the checkpoint files (without extension)

    Returns:
        Dictionary containing the statistics

    Raises:
        FileNotFoundError: If no .stats file exists at the base path
        ValueError: If the .stats file contains invalid JSON
    """
    base_path = Path(base_path)
    stats_path = base_path.with_suffix(".stats")

    if not stats_path.exists():
        raise FileNotFoundError(f"No statistics file found at {stats_path}")

    try:
        with open(stats_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in statistics file {stats_path}: {e}")


def from_path(path: str | Path) -> PolicyAgent:
    """Load a PolicyAgent from a file path.

    This function automatically determines the file type and loads accordingly:
    - If path ends in .pt: loads using torch.load and returns the agent
    - If path ends in .safetensors: loads weights and looks for .init_config file
    - If path has no extension: tries .safetensors first, then .pt

    Args:
        path: Path to the file or base path (without extension)

    Returns:
        PolicyAgent loaded from the path

    Raises:
        FileNotFoundError: If no valid files found at the path
        ValueError: If file format is invalid or incompatible
    """
    path = Path(path)

    # Handle specific file extensions
    if path.suffix == ".pt":
        agent = torch.load(path, weights_only=False)
        return agent

    elif path.suffix == ".safetensors":
        # Load weights from safetensors
        try:
            from safetensors.torch import load_file
            weights = load_file(str(path))
        except ImportError:
            raise ImportError(f"Required safetensors file not found at {path}")

        # Look for corresponding .init_config file
        init_config_path = path.with_suffix(".init_config")
        if not init_config_path.exists():
            raise FileNotFoundError(f"Required .init_config file not found at {init_config_path}")

        with open(init_config_path, "r") as f:
            init_config_data = json.load(f)
            from metta.agent.agent_env_config import AgentEnvConfig
            init_config = AgentEnvConfig.model_validate(init_config_data)

        return from_weights(weights, init_config)

    else:
        # Try safetensors first, then pt
        safetensors_path = path.with_suffix(".safetensors")
        if safetensors_path.exists():
            return from_path(safetensors_path)

        pt_path = path.with_suffix(".pt")
        if pt_path.exists():
            return from_path(pt_path)

        raise FileNotFoundError(f"No valid policy files found at {path}")
