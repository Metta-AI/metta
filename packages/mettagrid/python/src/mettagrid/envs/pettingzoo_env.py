"""
MettaGridPettingZooEnv - PettingZoo adapter for MettaGrid.

This class implements the PettingZoo ParallelEnv interface using the base MettaGridEnv.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from typing_extensions import override

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.core import MettaGridCore

# Data types for PettingZoo - import from C++ module
from mettagrid.mettagrid_c import (
    dtype_actions,
)


class MettaGridPettingZooEnv(MettaGridCore, ParallelEnv):
    """
    PettingZoo ParallelEnv adapter for MettaGrid environments.

    This class provides a PettingZoo-compatible interface for MettaGrid environments,
    using the parallel environment API for multi-agent scenarios.
    No training features are included - this is purely for PettingZoo compatibility.

    Inherits from:
    - MettaGridCore: Core C++ environment wrapper functionality
    - pettingzoo.ParallelEnv: Parallel multi-agent environment interface
      https://github.com/Farama-Foundation/PettingZoo/blob/405e71c912dc3f787bb12c7f8463f18fcce31bb1/pettingzoo/utils/env.py#L281
    """

    def __init__(
        self,
        mg_config: MettaGridConfig,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize PettingZoo environment.

        Args:
            mg_config: Environment configuration
            render_mode: Rendering mode
            **kwargs: Additional arguments
        """

        # Initialize core environment (no training features)
        MettaGridCore.__init__(
            self,
            mg_config,
            render_mode=render_mode,
        )

        # PettingZoo attributes
        self.agents: List[str] = []
        self.possible_agents: List[str] = []

        # populate possible_agents immediately (PettingZoo spec)
        num_agents = self.num_agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]

        # Create space objects once to avoid memory leaks
        # PettingZoo requires same space object instances to be returned
        self._observation_space_obj = self._observation_space
        self._action_space_obj = self._action_space

    def _setup_agents(self) -> None:
        """Setup agent names after core environment is created."""
        # Create agent names - c_env property handles the None check
        num_agents = self.num_agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()

    # Buffer management is handled by base MettaGridEnv class

    @override  # pettingzoo.ParallelEnv.reset
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Additional options (unused)

        Returns:
            Tuple of (observations_dict, infos_dict)
        """
        del options  # Unused parameter

        # Call the base reset method
        obs_array, info = super().reset(seed)

        # Setup agents if not already done
        if not self.agents:
            self._setup_agents()

        # Convert to PettingZoo format
        observations = {agent: obs_array[i] for i, agent in enumerate(self.agents)}
        infos = {agent: info for agent in self.agents}

        return observations, infos

    @override  # pettingzoo.ParallelEnv.step
    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict[str, Any]]]:
        """
        Execute one timestep of the environment dynamics.

        Args:
            actions: Dictionary mapping agent names to actions

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos)
        """
        # Convert actions dict to array format
        actions_array = np.zeros((len(self.agents), 2), dtype=dtype_actions)
        for i, agent in enumerate(self.agents):
            if agent in actions:
                actions_array[i] = actions[agent].astype(dtype_actions)

        # Call base step implementation
        observations, rewards, terminals, truncations, infos = super().step(actions_array)

        # Step results are already provided by base class

        # Convert to PettingZoo format
        obs_dict = {agent: observations[i] for i, agent in enumerate(self.agents)}
        reward_dict = {agent: float(rewards[i]) for i, agent in enumerate(self.agents)}
        terminal_dict = {agent: bool(terminals[i]) for i, agent in enumerate(self.agents)}
        truncation_dict = {agent: bool(truncations[i]) for i, agent in enumerate(self.agents)}
        info_dict = {agent: infos for agent in self.agents}

        # Remove agents that are done
        active_agents = []
        for agent in self.agents:
            if not (terminal_dict[agent] or truncation_dict[agent]):
                active_agents.append(agent)

        # Update agents list
        self.agents = active_agents

        return obs_dict, reward_dict, terminal_dict, truncation_dict, info_dict

    # PettingZoo required methods
    @override  # pettingzoo.ParallelEnv.observation_space
    def observation_space(self, agent: str) -> spaces.Box:
        """Get observation space for a specific agent."""
        del agent  # Unused parameter - all agents have same space
        # Return the same space object instance (PettingZoo requirement)
        return self._observation_space_obj

    @override  # pettingzoo.ParallelEnv.action_space
    def action_space(self, agent: str) -> spaces.MultiDiscrete:
        """Get action space for a specific agent."""
        del agent  # Unused parameter - all agents have same space
        # Return the same space object instance (PettingZoo requirement)
        return self._action_space_obj

    @override  # pettingzoo.ParallelEnv.state
    def state(self) -> np.ndarray:
        """
        Get global state (optional for PettingZoo).

        Returns:
            Global state array
        """
        # For state, we can return a flattened representation of all current observations
        # Since we don't store observations, we'll create a zero state of appropriate size
        obs_space = self._observation_space_obj
        total_size = self.num_agents * int(np.prod(obs_space.shape))
        return np.zeros(total_size, dtype=obs_space.dtype)

    @property
    def state_space(self) -> spaces.Box:
        """Get state space (optional for PettingZoo)."""
        # State space is flattened observation space
        obs_space = self._observation_space_obj
        total_size = self.num_agents * int(np.prod(obs_space.shape))

        return spaces.Box(
            low=obs_space.low.flatten()[0],
            high=obs_space.high.flatten()[0],
            shape=(total_size,),
            dtype=obs_space.dtype.type,
        )

    @override  # pettingzoo.ParallelEnv.render
    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment."""
        del mode  # Unused parameter
        return super().render()

    @override  # pettingzoo.ParallelEnv.close
    def close(self) -> None:
        """Close the environment."""
        super().close()

    @property
    def max_num_agents(self) -> int:
        """Get maximum number of agents."""
        return len(self.possible_agents)

    # num_agents property is defined above
