"""Policy artifact representation for Metta RL framework.

This module provides the PolicyArtifact class which can represent either a MettaAgent
or weights+statistics, with conversion methods between the two representations.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from metta.agent.metta_agent import MettaAgent, PolicyAgent
from metta.mettagrid import MettaGridEnv
from metta.agent.agent_config import AgentConfig
from metta.agent.agent_env_config import AgentEnvConfig


class PolicyArtifact:
    """Represents a policy artifact that can be either a MettaAgent or weights+statistics.

    This class provides a unified interface for handling different types of policy representations
    with explicit fields for each case.
    """

    def __init__(
        self,
        agent: Optional[Any] = None,
        weights: Optional[dict[str, torch.Tensor]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        init_config: Optional[AgentEnvConfig] = None,
    ):
        """Initialize a PolicyArtifact.

        Args:
            agent: The MettaAgent object (if representing an agent)
            weights: Dictionary of weights (if representing weights)
            statistics: Policy statistics
        """
        if agent is not None and weights is not None:
            raise ValueError("Cannot specify both agent and weights")

        if agent is None and weights is None:
            raise ValueError("One of weights or agent must be not-None")

        if weights is not None and statistics is None:
            raise ValueError("Metadata should only exist with weights")

        self.agent = agent
        self.weights = weights
        self.statistics = statistics or {}
        self.init_config = init_config

    @classmethod
    def write_agent_to_file(cls, agent: Any, write_to: str) -> None:
        base_path = Path(write_to)
        # Write agent to .pt file via pickling
        pt_path = base_path.with_suffix(".pt")
        torch.save(agent, pt_path)

    @classmethod
    def from_weights(cls, weights: Dict[str, torch.Tensor], statistics: Optional[Dict[str, Any]], init_config: AgentEnvConfig,
        write_to: Optional[str] = None) -> "PolicyAgent":
        """Create a PolicyArtifact from weights and statistics.

        Args:
            weights: Dictionary mapping parameter names to tensors
            statistics: Policy statistics

        Returns:
            PolicyArtifact with weights and statistics fields set
        """
        if not isinstance(weights, dict):
            raise ValueError("Weights must be a dictionary")

        # Validate that all values are tensors
        for key, value in weights.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"All values must be tensors, got {type(value)} for key {key}")

        artifact = cls(weights=weights, statistics=statistics, init_config=init_config)
        if write_to is not None:
            base_path = Path(write_to)
            # Write weights to .safetensors and statistics to .stats
            if artifact.weights is None:
                raise ValueError("Cannot write weights: weights are None")

            # Write weights (fallback to .pt if safetensors not available)
            try:
                from safetensors.torch import save_file

                safetensors_path = base_path.with_suffix(".safetensors")
                save_file(artifact.weights, str(safetensors_path))
            except ImportError:
                # Fallback to torch.save if safetensors not available
                # ?? do not do this!
                pt_path = base_path.with_suffix(".pt")
                torch.save(artifact.weights, pt_path)

            # Write statistics to .stats file
            stats_path = base_path.with_suffix(".stats")
            with open(stats_path, "w") as f:
                json.dump(artifact.statistics, f, indent=2)

            # Write init config to .init_config file
            if artifact.init_config is not None:
                init_config_path = base_path.with_suffix(".init_config")
                with open(init_config_path, "w") as f:
                    f.write(artifact.init_config.model_dump_json(indent=2))

        # Create a PolicyAgent from the weights and statistics
        return MettaAgent(artifact.init_config, AgentConfig.model_validate(artifact.init_config.agent_config))




    def get_statistics(self) -> Dict[str, Any]:
        """Get the statistics if this artifact contains weights.

        Returns:
            Policy statistics

        Raises:
            ValueError: If this artifact doesn't contain weights
        """
        if not (self.weights is not None):
            raise ValueError("This artifact contains an agent, not weights")
        return self.statistics

    def to_agent(self, env: "MettaGridEnv", cfg: "AgentConfig") -> Any:
        """Convert to agent representation.

        If this is already an agent, return it. If this is weights,
        create a new agent instance.

        Args:
            agent_class: Class to instantiate (required if converting from weights)
            **kwargs: Additional arguments to pass to agent constructor

        Returns:
            The agent object

        Raises:
            ValueError: If converting from weights without providing agent_class
        """
        if self.agent is not None:
            return self.agent

        if self.weights is None:
            raise ValueError("Weights are required to convert to an agent")

        agent = MettaAgent.from_weights(self.weights, env, cfg)
        return agent



    @classmethod
    def get_statistics_from_path(cls, base_path: str | Path) -> Dict[str, Any]:
        """Get statistics from a checkpoint base path without loading the full artifact.

        This method is useful when you only need the statistics/metadata and don't want
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

    @classmethod
    def from_path(cls, path: str | Path) -> "PolicyAgent":
        """Load a PolicyArtifact from a file path.

        This method automatically determines the file type and loads accordingly:
        - If path ends in .pt: loads using torch.load and returns from_agent
        - If path ends in .safetensors: loads weights and looks for .stats file
        - If path has no extension: tries .safetensors first, then .pt

        Args:
            path: Path to the file or base path (without extension)

        Returns:
            PolicyArtifact loaded from the path

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
                # Fallback to torch.load if safetensors not available
                weights = torch.load(path, weights_only=True)

            # Look for corresponding .stats and .init_config files
            stats_path = path.with_suffix(".stats")
            init_config_path = path.with_suffix(".init_config")

            statistics = {}
            if stats_path.exists():
                with open(stats_path, "r") as f:
                    statistics = json.load(f)

            if not init_config_path.exists():
                raise FileNotFoundError(f"Required .init_config file not found at {init_config_path}")

            with open(init_config_path, "r") as f:
                init_config_data = json.load(f)
                from metta.agent.agent_env_config import AgentEnvConfig
                init_config = AgentEnvConfig.model_validate(init_config_data)

            return cls.from_weights(weights, statistics, init_config)

        else:
            # Try safetensors first, then pt
            safetensors_path = path.with_suffix(".safetensors")
            if safetensors_path.exists():
                return cls.from_path(safetensors_path)

            pt_path = path.with_suffix(".pt")
            if pt_path.exists():
                return cls.from_path(pt_path)

            raise FileNotFoundError(f"No valid PolicyArtifact files found at {path}")
