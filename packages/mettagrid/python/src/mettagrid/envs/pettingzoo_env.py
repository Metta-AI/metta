"""
MettaGridPettingZooEnv - PettingZoo adapter for MettaGrid.

This class implements the PettingZoo ParallelEnv interface using the Simulator/Simulation API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from typing_extensions import override

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.mettagrid_c import dtype_actions
from mettagrid.simulator import Simulation, Simulator


class MettaGridPettingZooEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv adapter for MettaGrid environments.

    This class provides a PettingZoo-compatible interface for MettaGrid environments,
    using the parallel environment API for multi-agent scenarios.
    No training features are included - this is purely for PettingZoo compatibility.

    Implements:
    - pettingzoo.ParallelEnv: Parallel multi-agent environment interface
      https://github.com/Farama-Foundation/PettingZoo/blob/405e71c912dc3f787bb12c7f8463f18fcce31bb1/pettingzoo/utils/env.py#L281
    """

    def __init__(self, simulator: Simulator, cfg: MettaGridConfig, **kwargs: Any):
        """
        Initialize PettingZoo environment.

        Args:
            simulator: Simulator instance
            cfg: MettaGridConfig instance
            **kwargs: Additional arguments
        """
        super().__init__()

        self._simulator = simulator
        self._cfg = cfg
        self._seed = 0

        # Initialize first simulation to get space information
        self._sim: Simulation | None = None
        self._sim = self._simulator.new_simulation(cfg, seed=self._seed)
        assert self._sim is not None

        # PettingZoo attributes - agent IDs are integers
        self.possible_agents: List[int] = list(range(self._sim.num_agents))
        self.agents: List[int] = self.possible_agents.copy()

        # Create space objects once to avoid memory leaks
        # PettingZoo requires same space object instances to be returned
        obs_shape = self._sim.observation_shape
        self._observation_space_obj = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self._action_space_obj = spaces.Discrete(len(self._sim.action_names))

    @override
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Additional options (unused)

        Returns:
            Tuple of (observations_dict, infos_dict)
        """
        # Close current simulation if it exists
        if self._sim is not None:
            self._sim.close()

        # Update seed if provided
        if seed is not None:
            self._seed = seed

        # Create new simulation
        self._sim = self._simulator.new_simulation(self._cfg, seed=self._seed)
        assert self._sim is not None

        # Reset agents list to all possible agents
        self.agents = self.possible_agents.copy()

        # Get observations from C++ simulation
        obs_array = self._sim._c_sim.observations()

        # Convert to PettingZoo format (dict keyed by agent ID)
        observations = {agent_id: obs_array[agent_id] for agent_id in self.agents}
        infos = {agent_id: {} for agent_id in self.agents}

        return observations, infos

    @override  # pettingzoo.ParallelEnv.step
    def step(
        self, actions: Dict[int, np.ndarray | int]
    ) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[int, bool],
        Dict[int, Dict[str, Any]],
    ]:
        """
        Execute one timestep of the environment dynamics.

        Args:
            actions: Dictionary mapping agent IDs (int) to actions

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos)
        """
        sim = self._sim
        assert sim is not None
        # Set actions for all agents through C++ environment
        action_array = sim._c_sim.actions()
        for agent_id in self.agents:
            if agent_id in actions:
                action_idx = int(np.asarray(actions[agent_id], dtype=dtype_actions).reshape(()).item())
                action_array[agent_id] = action_idx

        # Execute simulation step
        sim.step()

        # Get results from C++ environment
        observations = sim._c_sim.observations()
        rewards = sim._c_sim.rewards()
        terminals = sim._c_sim.terminals()
        truncations = sim._c_sim.truncations()

        # Convert to PettingZoo format (dict keyed by agent ID)
        obs_dict = {agent_id: observations[agent_id] for agent_id in self.agents}
        reward_dict = {agent_id: float(rewards[agent_id]) for agent_id in self.agents}
        terminal_dict = {agent_id: bool(terminals[agent_id]) for agent_id in self.agents}
        truncation_dict = {agent_id: bool(truncations[agent_id]) for agent_id in self.agents}
        info_dict = {agent_id: {} for agent_id in self.agents}

        # Remove agents that are done
        active_agents = []
        for agent_id in self.agents:
            if not (terminal_dict[agent_id] or truncation_dict[agent_id]):
                active_agents.append(agent_id)

        # Update agents list
        self.agents = active_agents

        return obs_dict, reward_dict, terminal_dict, truncation_dict, info_dict

    # PettingZoo required methods
    @override  # pettingzoo.ParallelEnv.observation_space
    def observation_space(self, agent: int) -> spaces.Box:
        """Get observation space for a specific agent."""
        del agent  # Unused parameter - all agents have same space
        # Return the same space object instance (PettingZoo requirement)
        return self._observation_space_obj

    @override  # pettingzoo.ParallelEnv.action_space
    def action_space(self, agent: int) -> spaces.Discrete:
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
        total_size = len(self.possible_agents) * int(np.prod(obs_space.shape))
        return np.zeros(total_size, dtype=obs_space.dtype)

    @property
    def state_space(self) -> spaces.Box:
        """Get state space (optional for PettingZoo)."""
        # State space is flattened observation space
        obs_space = self._observation_space_obj
        total_size = len(self.possible_agents) * int(np.prod(obs_space.shape))

        return spaces.Box(
            low=obs_space.low.flatten()[0],
            high=obs_space.high.flatten()[0],
            shape=(total_size,),
            dtype=obs_space.dtype.type,
        )

    @override  # pettingzoo.ParallelEnv.render
    def render(self) -> None:
        """Render the environment (not implemented)."""
        pass

    @override  # pettingzoo.ParallelEnv.close
    def close(self) -> None:
        """Close the environment."""
        if self._sim is None:
            return
        self._sim.close()
        self._sim = None

    @property
    def max_num_agents(self) -> int:
        """Get maximum number of agents."""
        return len(self.possible_agents)

    @property
    def max_steps(self) -> int:
        """Get maximum number of steps before truncation."""
        return self._cfg.game.max_steps

    # num_agents property is defined above
