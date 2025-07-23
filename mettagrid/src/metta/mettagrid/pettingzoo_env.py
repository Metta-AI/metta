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

from metta.mettagrid.base_env import MettaGridEnv
from metta.mettagrid.curriculum import Curriculum
from metta.mettagrid.level_builder import Level
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter

# Data types for PettingZoo
dtype_observations = np.dtype(np.uint8)
dtype_terminals = np.dtype(bool)
dtype_truncations = np.dtype(bool)
dtype_rewards = np.dtype(np.float32)
dtype_actions = np.dtype(np.int32)


class MettaGridPettingZooEnv(MettaGridEnv, ParallelEnv):
    """
    PettingZoo ParallelEnv adapter for MettaGrid environments.

    This class provides a PettingZoo-compatible interface for MettaGrid environments,
    using the parallel environment API for multi-agent scenarios.
    """

    def __init__(
        self,
        curriculum: Curriculum,
        render_mode: Optional[str] = None,
        level: Optional[Level] = None,
        stats_writer: Optional[StatsWriter] = None,
        replay_writer: Optional[ReplayWriter] = None,
        is_training: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize PettingZoo environment.

        Args:
            curriculum: Curriculum for task management
            render_mode: Rendering mode
            level: Optional pre-built level
            stats_writer: Optional stats writer
            replay_writer: Optional replay writer
            is_training: Whether this is for training
            **kwargs: Additional arguments
        """
        # Initialize base environment
        MettaGridEnv.__init__(
            self,
            curriculum=curriculum,
            render_mode=render_mode,
            level=level,
            stats_writer=stats_writer,
            replay_writer=replay_writer,
            is_training=is_training,
            **kwargs,
        )

        # Create initial core environment for property access
        self._core_env = self._create_core_env(0)

        # PettingZoo attributes
        self.agents: List[str] = []
        self.possible_agents: List[str] = []

        # populate possible_agents immediately (PettingZoo spec)
        num_agents = self._core_env.num_agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]

        # Create space objects once to avoid memory leaks
        # PettingZoo requires same space object instances to be returned
        self._observation_space_obj = self._core_env.observation_space
        self._action_space_obj = self._core_env.action_space

        # Buffers for environment data
        self._observations: Optional[np.ndarray] = None
        self._terminals: Optional[np.ndarray] = None
        self._truncations: Optional[np.ndarray] = None
        self._rewards: Optional[np.ndarray] = None

    def _setup_agents(self) -> None:
        """Setup agent names after core environment is created."""
        if self._core_env is None:
            raise RuntimeError("Core environment not initialized")

        # Create agent names
        num_agents = self._core_env.num_agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()

    def _allocate_buffers(self) -> None:
        """Allocate buffers based on environment dimensions."""
        if self._core_env is None:
            raise RuntimeError("Core environment not initialized")

        num_agents = self._core_env.num_agents
        obs_space = self._core_env.observation_space

        # Allocate buffers
        self._observations = np.zeros((num_agents,) + obs_space.shape, dtype=dtype_observations)
        self._terminals = np.zeros(num_agents, dtype=dtype_terminals)
        self._truncations = np.zeros(num_agents, dtype=dtype_truncations)
        self._rewards = np.zeros(num_agents, dtype=dtype_rewards)

    @override
    def _get_initial_observations(self) -> np.ndarray:
        """
        Get initial observations and set up buffers.

        Returns:
            Initial observations array
        """
        if self._core_env is None:
            raise RuntimeError("Core environment not initialized")

        # Setup agents
        self._setup_agents()

        # Allocate buffers
        self._allocate_buffers()

        # Set buffers in core environment
        assert self._observations is not None
        assert self._terminals is not None
        assert self._truncations is not None
        assert self._rewards is not None
        self._core_env.set_buffers(self._observations, self._terminals, self._truncations, self._rewards)

        # Get initial observations
        return self._core_env.get_initial_observations()

    @override
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
        obs_array, info = self.reset_base(seed)

        # Convert to PettingZoo format
        observations = {agent: obs_array[i] for i, agent in enumerate(self.agents)}
        infos = {agent: info for agent in self.agents}

        return observations, infos

    @override
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
        infos = self.step_base(actions_array)

        # Get step results
        if self._observations is None or self._rewards is None or self._terminals is None or self._truncations is None:
            raise RuntimeError("Buffers not initialized")

        observations = self._observations.copy()
        rewards = self._rewards.copy()
        terminals = self._terminals.copy()
        truncations = self._truncations.copy()

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
    def observation_space(self, agent: str) -> spaces.Box:
        """Get observation space for a specific agent."""
        del agent  # Unused parameter - all agents have same space
        # Return the same space object instance (PettingZoo requirement)
        return self._observation_space_obj

    def action_space(self, agent: str) -> spaces.MultiDiscrete:
        """Get action space for a specific agent."""
        del agent  # Unused parameter - all agents have same space
        # Return the same space object instance (PettingZoo requirement)
        return self._action_space_obj

    def state(self) -> np.ndarray:
        """
        Get global state (optional for PettingZoo).

        Returns:
            Global state array
        """
        if self._observations is None:
            raise RuntimeError("Environment not initialized")

        # Return flattened observations as global state
        return self._observations.flatten()

    @property
    def state_space(self) -> spaces.Box:
        """Get state space (optional for PettingZoo)."""
        if self._core_env is None:
            raise RuntimeError("Environment not initialized")

        # State space is flattened observation space
        obs_space = self._core_env.observation_space
        total_size = self._core_env.num_agents * int(np.prod(obs_space.shape))

        return spaces.Box(
            low=obs_space.low.flatten()[0],
            high=obs_space.high.flatten()[0],
            shape=(total_size,),
            dtype=obs_space.dtype.type,
        )

    def render(self, mode: str = "human") -> Optional[str]:
        """Render the environment."""
        del mode  # Unused parameter
        return super().render()

    def close(self) -> None:
        """Close the environment."""
        super().close()

    @property
    def max_num_agents(self) -> int:
        """Get maximum number of agents."""
        return len(self.possible_agents)

    @property
    def num_agents(self) -> int:
        """Get current number of agents."""
        return len(self.agents)
