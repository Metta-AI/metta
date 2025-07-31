"""
MettaGridPettingZooEnv - PettingZoo adapter for MettaGrid.

This class implements the PettingZoo ParallelEnv interface using the base MettaGridEnv.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf
from pettingzoo import ParallelEnv
from typing_extensions import override

from metta.mettagrid.core import MettaGridCore
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.level_builder import Level

# Data types for PettingZoo - import from C++ module
from metta.mettagrid.mettagrid_c import (
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)


class MettaGridPettingZooEnv(MettaGridCore, ParallelEnv):
    """
    PettingZoo ParallelEnv adapter for MettaGrid environments.

    This class provides a PettingZoo-compatible interface for MettaGrid environments,
    using the parallel environment API for multi-agent scenarios.
    No training features are included - this is purely for PettingZoo compatibility.
    """

    def __init__(
        self,
        curriculum: Curriculum,
        render_mode: Optional[str] = None,
        level: Optional[Level] = None,
        **kwargs: Any,
    ):
        """
        Initialize PettingZoo environment.

        Args:
            curriculum: Curriculum for task management
            render_mode: Rendering mode
            level: Optional pre-built level
            **kwargs: Additional arguments
        """
        # Get level from curriculum if not provided
        if level is None:
            task = curriculum.get_task()
            level = task.env_cfg().game.map_builder.build()

        # Ensure we have a level
        assert level is not None, "Level must be provided or generated from curriculum"

        # Store curriculum for reset operations
        self._curriculum = curriculum
        self._task = self._curriculum.get_task()

        # Get game config for core initialization
        task_cfg = self._task.env_cfg()
        game_config_dict = OmegaConf.to_container(task_cfg.game)
        assert isinstance(game_config_dict, dict), "Game config must be a dictionary"

        # Initialize core environment (no training features)
        MettaGridCore.__init__(
            self,
            level=level,
            game_config_dict=game_config_dict,
            render_mode=render_mode,
            **kwargs,
        )

        # PettingZoo attributes
        self.agents: List[str] = []
        self.possible_agents: List[str] = []

        # populate possible_agents immediately (PettingZoo spec)
        num_agents = self.c_env.num_agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]

        # Create space objects once to avoid memory leaks
        # PettingZoo requires same space object instances to be returned
        self._observation_space_obj = self.c_env.observation_space
        self._action_space_obj = self.c_env.action_space

        # Initialize buffer attributes for memory management
        self._observations: Optional[np.ndarray] = None
        self._terminals: Optional[np.ndarray] = None
        self._truncations: Optional[np.ndarray] = None
        self._rewards: Optional[np.ndarray] = None

        # Allocate buffers for C++ environment
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """Allocate numpy arrays for C++ environment shared memory."""
        # Allocate observation buffer
        obs_shape = (self.num_agents, *self._observation_space.shape)
        self._observations = np.zeros(obs_shape, dtype=dtype_observations)

        # Allocate terminal/truncation/reward buffers
        self._terminals = np.zeros(self.num_agents, dtype=dtype_terminals)
        self._truncations = np.zeros(self.num_agents, dtype=dtype_truncations)
        self._rewards = np.zeros(self.num_agents, dtype=dtype_rewards)

        # Set buffers in C++ environment for direct writes
        self.c_env.set_buffers(self._observations, self._terminals, self._truncations, self._rewards)

    def _setup_agents(self) -> None:
        """Setup agent names after core environment is created."""
        # Create agent names - c_env property handles the None check
        num_agents = self.c_env.num_agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents.copy()

    # Buffer management is handled by base MettaGridEnv class

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

        # Get new task from curriculum and its config
        self._task = self._curriculum.get_task()
        task_cfg = self._task.env_cfg()
        game_config_dict = OmegaConf.to_container(task_cfg.game)
        assert isinstance(game_config_dict, dict), "Game config must be a dictionary"

        obs_array, info = super().reset(game_config_dict, seed)

        # Setup agents if not already done
        if not self.agents:
            self._setup_agents()

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
        # For state, we can return a flattened representation of all current observations
        # Since we don't store observations, we'll create a zero state of appropriate size
        obs_space = self.c_env.observation_space
        total_size = self.c_env.num_agents * int(np.prod(obs_space.shape))
        return np.zeros(total_size, dtype=obs_space.dtype)

    @property
    def state_space(self) -> spaces.Box:
        """Get state space (optional for PettingZoo)."""
        # State space is flattened observation space
        obs_space = self.c_env.observation_space
        total_size = self.c_env.num_agents * int(np.prod(obs_space.shape))

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

    # num_agents property is defined above
