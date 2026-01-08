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

import logging
from typing import Any, Dict, List, Optional, Tuple

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
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulation, Simulator
from mettagrid.simulator.simulator import Buffers
from pufferlib.pufferlib import PufferEnv  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Type compatibility assertions - ensure C++ types match PufferLib expectations
# PufferLib expects particular datatypes - see pufferlib/vector.py
assert dtype_observations == np.dtype(np.uint8)
assert dtype_terminals == np.dtype(np.bool_)
assert dtype_truncations == np.dtype(np.bool_)
assert dtype_rewards == np.dtype(np.float32)
assert dtype_actions == np.dtype(np.int32)


class MettaGridPufferEnv(PufferEnv):
    """
    Wraps the Simulator class to provide PufferLib compatibility.

    Inherits from pufferlib.PufferEnv: High-performance vectorized environment interface
      https://github.com/PufferAI/PufferLib/blob/main/pufferlib/environments.py
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        simulator: Simulator,
        cfg: MettaGridConfig,
        supervisor_policy_spec: Optional[PolicySpec] = None,
        buf: Any = None,
        seed: int = 0,
    ):
        # Support both Simulation and MettaGridConfig for backwards compatibility
        self._simulator = simulator
        self._current_cfg = cfg
        self._current_seed = seed
        self._supervisor_policy_spec = supervisor_policy_spec

        # Initialize shared buffers FIRST (before super().__init__)
        # because PufferLib may access them during initialization

        policy_env_info = PolicyEnvInterface.from_mg_cfg(cfg)

        self._buffers: Buffers = Buffers(
            observations=np.zeros(
                (policy_env_info.num_agents, *policy_env_info.observation_space.shape),
                dtype=dtype_observations,
            ),
            terminals=np.zeros(policy_env_info.num_agents, dtype=dtype_terminals),
            truncations=np.zeros(policy_env_info.num_agents, dtype=dtype_truncations),
            rewards=np.zeros(policy_env_info.num_agents, dtype=dtype_rewards),
            masks=np.ones(policy_env_info.num_agents, dtype=dtype_masks),
            actions=np.zeros(policy_env_info.num_agents, dtype=dtype_actions),
            teacher_actions=np.zeros(policy_env_info.num_agents, dtype=dtype_actions),
        )

        # Set observation and action spaces BEFORE calling super().__init__()
        # PufferLib requires these to be set first
        self.single_observation_space: Box = policy_env_info.observation_space
        self.single_action_space: Discrete = policy_env_info.action_space

        self._env_supervisor: MultiAgentPolicy | None = None
        self._sim: Optional[Simulation] = None
        self._sim = self._init_simulation()
        self.num_agents = self._sim.num_agents

        super().__init__(buf=buf)

    @property
    def env_cfg(self) -> MettaGridConfig:
        """Get the environment configuration."""
        return self._current_cfg

    def set_mg_config(self, config: MettaGridConfig) -> None:
        self._current_cfg = config

    def get_episode_rewards(self) -> np.ndarray:
        sim = self._sim
        assert sim is not None
        return sim.episode_rewards

    @property
    def current_simulation(self) -> Simulation:
        if self._sim is None:
            raise RuntimeError("Simulation is closed")
        return self._sim

    def _init_simulation(self) -> Simulation:
        sim = self._simulator.new_simulation(self._current_cfg, self._current_seed, buffers=self._buffers)
        if self._supervisor_policy_spec is not None:
            self._env_supervisor = initialize_or_load_policy(
                PolicyEnvInterface.from_mg_cfg(self._current_cfg),
                self._supervisor_policy_spec,
            )
            self._compute_supervisor_actions()
        return sim

    def _new_sim(self) -> None:
        if self._sim is not None:
            self._sim.close()
        self._sim = self._init_simulation()

    @override
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._current_seed = seed

        self._new_sim()

        return self._buffers.observations, {}

    @override
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        sim = self._sim
        assert sim is not None
        if sim._c_sim.terminals().all() or sim._c_sim.truncations().all():
            self._new_sim()
            sim = self._sim
            assert sim is not None

        # Gymnasium returns int64 arrays by default when sampling MultiDiscrete spaces,
        # so coerce here to keep callers simple while preserving strict bounds checking.
        actions_to_copy = actions if actions.dtype == dtype_actions else np.asarray(actions, dtype=dtype_actions)
        np.copyto(self._buffers.actions, actions_to_copy, casting="safe")

        sim.step()

        # Do this after step() so that the trainer can use it if needed
        if self._supervisor_policy_spec is not None:
            self._compute_supervisor_actions()

        return (
            self._buffers.observations,
            self._buffers.rewards,
            self._buffers.terminals,
            self._buffers.truncations,
            sim._context.get("infos", {}),
        )

    def _compute_supervisor_actions(self) -> None:
        supervisor = self._env_supervisor
        assert supervisor is not None
        teacher_actions = self._buffers.teacher_actions
        raw_observations = self._buffers.observations
        supervisor.step_batch(raw_observations, teacher_actions)

    def disable_supervisor(self) -> None:
        """Disable supervisor policy to avoid extra forward passes after teacher phase."""
        self._supervisor_policy_spec = None
        self._env_supervisor = None

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

    @property
    def teacher_actions(self) -> np.ndarray:
        return self._buffers.teacher_actions

    @teacher_actions.setter
    def teacher_actions(self, teacher_actions: np.ndarray) -> None:
        self._buffers.teacher_actions = teacher_actions

    @property
    def render_mode(self) -> str:
        """PufferLib render mode - returns 'ansi' for text-based rendering."""
        return "ansi"

    def render(self) -> str:
        """Render the current state as unicode text."""
        from mettagrid.renderer.miniscope.buffer import MapBuffer
        from mettagrid.renderer.miniscope.symbol import DEFAULT_SYMBOL_MAP

        sim = self._sim
        assert sim is not None
        symbol_map = DEFAULT_SYMBOL_MAP.copy()
        for obj in self._current_cfg.game.objects.values():
            if obj.render_name:
                symbol_map[obj.render_name] = obj.render_symbol
            symbol_map[obj.name] = obj.render_symbol

        return MapBuffer(
            symbol_map=symbol_map,
            initial_height=sim.map_height,
            initial_width=sim.map_width,
        ).render_full_map(sim._c_sim.grid_objects())

    def close(self) -> None:
        """Close the environment."""
        if self._sim is None:
            return
        self._sim.close()
        self._sim = None
