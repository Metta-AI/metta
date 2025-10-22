"""MettaGridCore - Core Python wrapper for MettaGrid C++ environment.

This class provides the base functionality for all framework-specific adapters,
without any training-specific features or framework dependencies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
from gymnasium.spaces import Box, Discrete

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.core import EpisodeStats
from mettagrid.mettagrid_c import MettaGrid as MettaGridCpp
from mettagrid.mettagrid_c import (
    dtype_actions,
    dtype_masks,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from mettagrid.profiling.stopwatch import Stopwatch, with_instance_timer


@dataclass
class ObservationFeature:
    feature_id: int
    name: str
    normalization: float


@dataclass
class Observation:
    feature: ObservationFeature
    location: tuple[int, int]
    value: int

    def row(self) -> int:
        return self.location[1]

    def col(self) -> int:
        return self.location[0]


class Action:
    action_id: int
    action_arg: int


class MoveAction(Action):
    """Move action."""

    direction: Literal["NW", "N", "NE", "W", "E", "SW", "S", "SE"]


class NoopAction(Action):
    """Noop action."""

    pass


logger = logging.getLogger("Simulator")


@dataclass
class BoundingBox:
    min_row: int
    max_row: int
    min_col: int
    max_col: int


class SimulatorEventHandler:
    """Handler for Simulator events."""

    def __init__(self, simulator: "Simulator"):
        self._simulator = simulator

    def on_episode_start(self) -> None:
        pass

    def on_episode_end(self) -> None:
        pass

    def on_step(self) -> None:
        pass

    def on_close(self) -> None:
        pass


@dataclass
class Buffers:
    observations: np.ndarray
    terminals: np.ndarray
    truncations: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    actions: np.ndarray


class Simulator:
    """
    Simulator for MettaGrid environments.

    This class provides a Python interface to the MettaGrid C++ environment.
    """

    def __init__(self, config: MettaGridConfig):
        """Initialize the simulator."""

        self._config = config
        self._current_seed: int = 0
        self._map_builder = self._config.game.map_builder.create()

        self._observation_space = Box(0, 255, (200, 3), dtype=dtype_observations)  # xcxc
        self._action_space = Discrete(self._c_sim.num_actions())  # xcxc

        # Buffers for fast data access
        self._buffers: Buffers = Buffers(
            observations=np.zeros((self.num_agents, *self._observation_space.shape), dtype=dtype_observations),
            terminals=np.zeros(self.num_agents, dtype=dtype_terminals),
            truncations=np.zeros(self.num_agents, dtype=dtype_truncations),
            rewards=np.zeros(self.num_agents, dtype=dtype_rewards),
            masks=np.ones(self.num_agents, dtype=dtype_masks),
            actions=np.zeros(self.num_agents, dtype=dtype_actions),
        )
        self._c_sim: MettaGridCpp = self._create_c_sim()
        self._update_buffers()

        self._event_handlers: List[SimulatorEventHandler] = []

        self.timer = Stopwatch(log_level=logger.getEffectiveLevel())
        self.timer.start()
        self.timer.start("thread_idle")

    @with_instance_timer("reset")
    def reset(self, config: Optional[MettaGridConfig] = None, seed: Optional[int] = None) -> None:
        """Reset the simulator configuration with new config and seed."""
        if config is not None:
            self._config = config
        if seed is not None:
            self._current_seed = seed
        self._map_builder = self._config.game.map_builder.create()
        self._c_sim = self._create_c_sim()
        self._update_buffers()

    def add_event_handler(self, handler: SimulatorEventHandler) -> None:
        self._event_handlers.append(handler)

    @with_instance_timer("start_episode")
    def start_episode(self) -> None:
        self._c_sim.start_episode()
        for handler in self._event_handlers:
            handler.on_episode_start()
        self.timer.start("thread_idle")

    def step(self, actions: list[Action]) -> Sequence[Observation]:
        arr = np.asarray(actions, dtype=dtype_actions)
        if arr.ndim != 1 or arr.shape[0] != self.num_agents:
            raise ValueError(
                f"Expected actions of shape ({self.num_agents},) but received {arr.shape}; "
                "ensure policies emit a scalar action id per agent"
            )
        return self._c_sim.step(arr)

    @with_instance_timer("step")
    def step_raw(self, actions: np.ndarray) -> None:
        """Execute one timestep of the environment dynamics with the given actions."""
        self.timer.stop("thread_idle")

        arr = np.asarray(actions, dtype=dtype_actions)
        if arr.ndim != 1 or arr.shape[0] != self.num_agents:
            raise ValueError(
                f"Expected actions of shape ({self.num_agents},) but received {arr.shape}; "
                "ensure policies emit a scalar action id per agent"
            )

        with self.timer("c_sim.step"):
            self._c_sim.step(arr)

        for handler in self._event_handlers:
            with self.timer(f"sim.on_step.{handler.__class__.__name__}"):
                handler.on_step()

        if self._c_sim.done():
            self.timer.start("episode_end")
            for handler in self._event_handlers:
                with self.timer(f"sim.on_episode_end.{handler.__class__.__name__}"):
                    handler.on_episode_end()
            self.timer.stop("episode_end")

        self.timer.start("thread_idle")

    def end_episode(self) -> None:
        """End the current episode."""
        assert self._buffers is not None
        self._buffers.truncations[:] = True

    def close(self) -> None:
        """Close the environment."""
        for handler in self._event_handlers:
            handler.on_close()
        del self._c_sim

    @property
    def config(self) -> MettaGridConfig:
        return self._config

    @property
    def episode_rewards(self) -> np.ndarray:
        """Get the episode rewards."""
        return self._c_sim.get_episode_rewards()

    @property
    def episode_stats(self) -> EpisodeStats:
        return self._c_sim.get_episode_stats()

    @property
    def current_step(self) -> int:
        return self._c_sim.current_step

    @property
    def num_agents(self) -> int:
        return self._c_sim.num_agents

    @property
    def action_names(self) -> List[str]:
        return self._c_sim.action_names()

    @property
    def object_type_names(self) -> List[str]:
        return self._c_sim.object_type_names()

    @property
    def resource_names(self) -> List[str]:
        return self._c_sim.resource_names()

    @property
    def feature_normalizations(self) -> Dict[int, float]:
        """Get feature normalizations from C++ environment."""
        # Check if the C++ environment has the direct method
        if hasattr(self._c_sim, "feature_normalizations"):
            return self._c_sim.feature_normalizations()
        else:
            # Fallback to extracting from feature_spec (slower)
            feature_spec = self._c_sim.feature_spec()
            return {int(spec["id"]): float(spec["normalization"]) for spec in feature_spec.values()}

    @property
    def initial_grid_hash(self) -> int:
        return self._c_sim.initial_grid_hash

    @property
    def action_success(self) -> List[bool]:
        return self._c_sim.action_success()

    @property
    def observation_features(self) -> Dict[str, ObsFeature]:
        """Build the features dictionary for initialize_to_environment."""
        # Get feature spec from C++ environment
        feature_spec = self._c_sim.feature_spec()

        features = {}
        for feature_name, feature_info in feature_spec.items():
            feature = ObsFeature(
                id=int(feature_info["id"]), normalization=feature_info["normalization"], name=feature_name
            )
            features[feature_name] = feature

        return features

    def grid_objects(
        self, bbox: Optional[BoundingBox] = None, ignore_types: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Get grid objects information, optionally filtered by bounding box and type.

        Args:
            bbox: Bounding box, None for no limit
            ignore_types: List of type names to exclude from results (e.g., ["wall"])

        Returns:
            Dictionary mapping object IDs to object dictionaries
        """
        if bbox is None:
            bbox = BoundingBox(min_row=-1, max_row=-1, min_col=-1, max_col=-1)

        ignore_list = ignore_types if ignore_types is not None else []
        return self._c_sim.grid_objects(bbox.min_row, bbox.max_row, bbox.min_col, bbox.max_col, ignore_list)

    def set_inventory(self, agent_id: int, inventory: Dict[str, int]) -> None:
        """Set an agent's inventory by resource name.

        Any resources not mentioned will be cleared in the underlying C++ call.
        """
        if not isinstance(agent_id, int):
            raise TypeError("agent_id must be an int")
        if not isinstance(inventory, dict):
            raise TypeError("inventory must be a dict[str, int]")

        # Build mapping from resource name to id
        name_to_id = {name: idx for idx, name in enumerate(self.resource_names)}

        # Convert names to ids, validating inputs
        inv_by_id: Dict[int, int] = {}
        for name, amount in inventory.items():
            if name not in name_to_id:
                raise KeyError(f"Unknown resource name: {name}")
            if not isinstance(amount, (int, np.integer)):
                raise TypeError(f"Amount for {name} must be int")
            inv_by_id[int(name_to_id[name])] = int(amount)

        # Forward to C++ binding
        self._c_sim.set_inventory(agent_id, inv_by_id)

    def _create_c_sim(self) -> MettaGridCpp:
        game_map = self._map_builder.build()

        # Handle spawn points: treat them as potential spawn locations
        # If there are more spawn points than agents, replace the excess with empty spaces
        spawn_mask = np.char.startswith(game_map.grid, "agent")
        level_agents = np.count_nonzero(spawn_mask)
        num_agents = self._config.game.num_agents

        if level_agents < num_agents:
            raise ValueError(
                f"Number of agents {num_agents} exceeds available spawn points {level_agents} in map. "
                f"This may be because your map, after removing border width, is too small to fit the number of agents."
            )
        elif level_agents > num_agents:
            # Replace excess spawn points with empty spaces
            spawn_indices = np.argwhere(spawn_mask)
            # Keep first num_agents spawn points, replace the rest with empty
            for idx in spawn_indices[num_agents:]:
                game_map.grid[tuple(idx)] = "empty"
        game_config_dict = self._config.game.model_dump()

        # Create C++ config
        try:
            c_cfg = from_mettagrid_config(game_config_dict)
        except Exception as e:
            logger.error(f"Error creating C++ config: {e}")
            logger.error(f"Game config: {game_config_dict}")
            raise e

        # Create C++ environment
        c_env = MettaGridCpp(c_cfg, game_map.grid.tolist(), self._current_seed)

        self._c_sim = c_env
        return c_env

    def _update_buffers(self) -> None:
        self._c_sim.set_buffers(
            self._buffers.observations,
            self._buffers.terminals,
            self._buffers.truncations,
            self._buffers.rewards,
        )
