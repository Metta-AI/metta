"""Policy artifact representation for Metta RL framework.

This module provides the PolicyArtifact class which can represent either a MettaAgent
or weights+statistics, with conversion methods between the two representations.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from metta.agent.metta_agent import MettaAgent
from metta.mettagrid import MettaGridEnv
from metta.agent.agent_config import AgentConfig


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

    @classmethod
    def from_agent(cls, agent: Any) -> "PolicyArtifact":
        """Create a PolicyArtifact from a MettaAgent.

        Args:
            agent: The MettaAgent object

        Returns:
            PolicyArtifact with agent field set
        """
        return cls(agent=agent)

    @classmethod
    def from_weights(cls, weights: Dict[str, torch.Tensor], statistics: Optional[Dict[str, Any]]) -> "PolicyArtifact":
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

        return cls(weights=weights, statistics=statistics)

    def is_agent(self) -> bool:
        """Check if this artifact contains an agent."""
        return self.agent is not None

    def is_weights(self) -> bool:
        """Check if this artifact contains weights."""
        return self.weights is not None

    def get_statistics(self) -> Dict[str, Any]:
        """Get the statistics if this artifact contains weights.

        Returns:
            Policy statistics

        Raises:
            ValueError: If this artifact doesn't contain weights
        """
        if not self.is_weights():
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
        if self.is_agent():
            return self.agent

        if self.weights is None:
            raise ValueError("Weights are required to convert to an agent")

        agent = MettaAgent.from_weights(self.weights, env, cfg)
        return agent

    def write_to_files(self, base_path: str | Path) -> None:
        """Write the artifact to files based on its type.

        Args:
            base_path: Base path for the output files (without extension)
        """
        base_path = Path(base_path)

        if self.is_agent():
            # Write agent to .pt file via pickling
            pt_path = base_path.with_suffix(".pt")
            torch.save(self.agent, pt_path)
        else:
            # Write weights to .safetensors and statistics to .stats
            if self.weights is None:
                raise ValueError("Cannot write weights: weights are None")

            # Write weights (fallback to .pt if safetensors not available)
            try:
                from safetensors.torch import save_file

                safetensors_path = base_path.with_suffix(".safetensors")
                save_file(self.weights, str(safetensors_path))
            except ImportError:
                # Fallback to torch.save if safetensors not available
                # ?? do not do this!
                pt_path = base_path.with_suffix(".pt")
                torch.save(self.weights, pt_path)

            # Write statistics to .stats file
            stats_path = base_path.with_suffix(".stats")
            with open(stats_path, "w") as f:
                json.dump(self.statistics, f, indent=2)

    @classmethod
    def read_from_files(cls, base_path: str | Path) -> "PolicyArtifact":
        """Read a PolicyArtifact from files.

        Args:
            base_path: Base path for the input files (without extension)

        Returns:
            PolicyArtifact loaded from files

        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If file format is invalid
        """
        base_path = Path(base_path)

        # Check if .pt file exists (agent case)
        pt_path = base_path.with_suffix(".pt")
        if pt_path.exists():
            agent = torch.load(pt_path, weights_only=False)
            return cls.from_agent(agent)

        # Check if .safetensors and .stats exist (weights case)
        safetensors_path = base_path.with_suffix(".safetensors")
        stats_path = base_path.with_suffix(".stats")

        if safetensors_path.exists() and stats_path.exists():
            # Load weights from safetensors
            try:
                from safetensors.torch import load_file
                weights = load_file(str(safetensors_path))
            except ImportError:
                # Fallback to torch.load if safetensors not available
                weights = torch.load(safetensors_path, weights_only=True)

            # Load statistics from .stats
            with open(stats_path, "r") as f:
                statistics = json.load(f)

            return cls.from_weights(weights, statistics)

        # Fallback: check for .pt file with weights only
        if pt_path.exists():
            weights = torch.load(pt_path, weights_only=True)
            return cls.from_weights(weights, {})

        raise FileNotFoundError(f"No valid PolicyArtifact files found at {base_path}")
