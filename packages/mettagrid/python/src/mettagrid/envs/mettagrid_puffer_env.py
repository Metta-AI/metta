"""
MettaGridPufferEnv - PufferLib integration for MettaGrid.

This class provides PufferLib compatibility for MettaGrid environments using
the Simulation class. This allows MettaGrid environments to be used
directly with PufferLib training infrastructure.

Provides:
 - Auto-reset on episode completion
 - Persistent buffers for re-use between resets

Architecture:
- MettaGridPufferEnv wraps Simulation and provides PufferEnv interface
- This enables MettaGridPufferEnv to work seamlessly with PufferLib training code

For users:
- Use MettaGridPufferEnv directly with PufferLib (it inherits PufferLib functionality)
- Alternatively, use PufferLib's MettaPuff wrapper for additional PufferLib features:
  https://github.com/PufferAI/PufferLib/blob/main/pufferlib/environments/metta/environment.py

This avoids double-wrapping while maintaining full PufferLib compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium.spaces import Box, Discrete
from typing_extensions import override

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.mettagrid_c import (
    dtype_actions,
    dtype_masks,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulation, Simulator
from pufferlib.pufferlib import PufferEnv

# Type compatibility assertions - ensure C++ types match PufferLib expectations
# PufferLib expects particular datatypes - see pufferlib/vector.py
assert dtype_observations == np.dtype(np.uint8)
assert dtype_terminals == np.dtype(np.bool_)
assert dtype_truncations == np.dtype(np.bool_)
assert dtype_rewards == np.dtype(np.float32)
assert dtype_actions == np.dtype(np.int32)


@dataclass
class Buffers:
    observations: np.ndarray
    terminals: np.ndarray
    truncations: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    actions: np.ndarray


class MettaGridPufferEnv(PufferEnv):
    """
    Wraps the Simulator class to provide PufferLib compatibility.

    Inherits from pufferlib.PufferEnv: High-performance vectorized environment interface
      https://github.com/PufferAI/PufferLib/blob/main/pufferlib/environments.py
    """

    def __init__(self, simulator: Simulator, cfg: MettaGridConfig, buf: Any = None, seed: int = 0):
        # Support both Simulation and MettaGridConfig for backwards compatibility
        self._simulator = simulator
        self._current_cfg = cfg
        self._current_seed = seed

        # Initialize shared buffers FIRST (before super().__init__)
        # because PufferLib may access them during initialization

        policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)

        self._buffers: Buffers = Buffers(
            observations=np.zeros(
                (policy_env_info.num_agents, *policy_env_info.observation_space.shape), dtype=dtype_observations
            ),
            terminals=np.zeros(policy_env_info.num_agents, dtype=dtype_terminals),
            truncations=np.zeros(policy_env_info.num_agents, dtype=dtype_truncations),
            rewards=np.zeros(policy_env_info.num_agents, dtype=dtype_rewards),
            masks=np.ones(policy_env_info.num_agents, dtype=dtype_masks),
            actions=np.zeros(policy_env_info.num_agents, dtype=dtype_actions),
        )

        # Set observation and action spaces BEFORE calling super().__init__()
        # PufferLib requires these to be set first
        self.single_observation_space: Box = policy_env_info.observation_space
        self.single_action_space: Discrete = policy_env_info.action_space

        self._new_sim()
        self.num_agents: int = self._sim.num_agents

        super().__init__(buf=buf)

    @property
    def env_cfg(self) -> MettaGridConfig:
        """Get the environment configuration."""
        return self._current_cfg

    def set_mg_config(self, config: MettaGridConfig) -> None:
        self._current_cfg = config

    def get_episode_rewards(self) -> np.ndarray:
        return self._sim.episode_rewards

    @property
    def current_simulation(self) -> Simulation:
        return self._sim

    def _update_buffers(self) -> None:
        """Set buffers on the C simulator."""
        self._sim._c_sim.set_buffers(
            self._buffers.observations,
            self._buffers.terminals,
            self._buffers.truncations,
            self._buffers.rewards,
            self._buffers.actions,
        )

    def _get_initial_observations(self) -> np.ndarray:
        observations, _ = super().reset()
        return observations

    def _new_sim(self) -> None:
        if hasattr(self, "_sim") and self._sim is not None:
            self._sim.close()
        self._sim = None
        self._sim = self._simulator.new_simulation(self._current_cfg, self._current_seed)
        self._update_buffers()
        self._buffers.rewards[:] = 0.0
        self._buffers.terminals[:] = False
        self._buffers.truncations[:] = False

    @override
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._current_seed = seed

        self._new_sim()
        return self._buffers.observations, {}

    @override
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        if self._sim._c_sim.terminals().all() or self._sim._c_sim.truncations().all():
            self._new_sim()

        self._buffers.actions[:] = actions
        self._sim._c_sim.actions()[:] = actions
        self._sim.step()

        return (
            self._buffers.observations,
            self._buffers.rewards,
            self._buffers.terminals,
            self._buffers.truncations,
            self._sim._context.get("infos", {}),
        )

    @property
    def observations(self) -> np.ndarray:
        return self._buffers.observations

    @observations.setter
    def observations(self, observations: np.ndarray) -> None:
        self._buffers.observations = observations

    @property
    def rewards(self) -> np.ndarray:
        return self._buffers.rewards

    @rewards.setter
    def rewards(self, rewards: np.ndarray) -> None:
        self._buffers.rewards = rewards

    @property
    def terminals(self) -> np.ndarray:
        return self._buffers.terminals

    @terminals.setter
    def terminals(self, terminals: np.ndarray) -> None:
        self._buffers.terminals = terminals

    @property
    def truncations(self) -> np.ndarray:
        return self._buffers.truncations

    @truncations.setter
    def truncations(self, truncations: np.ndarray) -> None:
        self._buffers.truncations = truncations

    @property
    def masks(self) -> np.ndarray:
        return self._buffers.masks

    @masks.setter
    def masks(self, masks: np.ndarray) -> None:
        self._buffers.masks = masks

    @property
    def actions(self) -> np.ndarray:
        return self._buffers.actions

    @actions.setter
    def actions(self, actions: np.ndarray) -> None:
        self._buffers.actions = actions

    def close(self) -> None:
        """Close the environment."""
        if hasattr(self, "_sim") and self._sim is not None:
            self._sim.close()
