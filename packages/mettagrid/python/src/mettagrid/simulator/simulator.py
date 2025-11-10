from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np

# Don't use `from ... import ...` here because it will cause a circular import.
import mettagrid.config.mettagrid_c_config as mettagrid_c_config
import mettagrid.config.mettagrid_config as mettagrid_config
from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.map_builder.map_builder import GameMap
from mettagrid.mettagrid_c import MettaGrid as MettaGridCpp
from mettagrid.mettagrid_c import PackedCoordinate
from mettagrid.profiling.stopwatch import Stopwatch, with_instance_timer
from mettagrid.simulator.interface import Action, AgentObservation, ObservationToken, SimulatorEventHandler

if TYPE_CHECKING:
    from mettagrid.mettagrid_c import EpisodeStats

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    min_row: int
    max_row: int
    min_col: int
    max_col: int


@dataclass
class Buffers:
    observations: np.ndarray
    terminals: np.ndarray
    truncations: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    teacher_actions: np.ndarray


class Simulation:
    def __init__(
        self,
        config: mettagrid_config.MettaGridConfig,
        seed: int = 0,
        event_handlers: Optional[Sequence[SimulatorEventHandler]] | None = None,
        simulator: Optional[Simulator] | None = None,
        buffers: Optional[Buffers] = None,
    ):
        self._config = config
        self._seed = seed
        self._event_handlers = event_handlers or []
        self._simulator = simulator
        self._context: Dict[str, Any] = {}

        for handler in self._event_handlers:
            handler.set_simulation(self)

        self._timer = Stopwatch(log_level=logger.getEffectiveLevel())
        self._timer.start()

        game_config_dict = self._config.game.model_dump()

        map_grid = self._make_map().grid.tolist()

        # Create C++ config
        try:
            c_cfg = mettagrid_c_config.convert_to_cpp_game_config(game_config_dict)
        except Exception as e:
            logger.error(f"Error creating C++ config: {e}")
            logger.error(f"Game config: {game_config_dict}")
            raise e

        # Create C++ environment
        self.__c_sim = MettaGridCpp(c_cfg, map_grid, self._seed)

        # Compute action_ids from config actions
        self._action_ids: dict[str, int] = {
            action.name: idx for idx, action in enumerate(self._config.game.actions.actions())
        }

        if buffers is not None:
            self._c_sim.set_buffers(
                buffers.observations,
                buffers.terminals,
                buffers.truncations,
                buffers.rewards,
                buffers.actions,
            )

        # Build feature dict from id_map
        self._features: dict[int, ObservationFeatureSpec] = {
            feature.id: feature for feature in self._config.id_map().features()
        }

        self._start_episode()

        self._timer.start("thread_idle")

    def agents(self) -> list[SimulationAgent]:
        return [self.agent(agent_id) for agent_id in range(self.num_agents)]

    def agent(self, agent_id: int) -> SimulationAgent:
        return SimulationAgent(self, agent_id)

    def observations(self) -> list[AgentObservation]:
        return [self.agent(agent_id).observation for agent_id in range(self.num_agents)]

    def is_done(self) -> bool:
        return bool(self.__c_sim.truncations().all() or self.__c_sim.terminals().all())

    def _start_episode(self) -> None:
        """Start a new episode (internal use only)."""
        self._episode_started = True
        self._context = {}

        for handler in self._event_handlers:
            with self._timer(f"sim.on_episode_start.{handler.__class__.__name__}"):
                handler.on_episode_start()

    def end_episode(self) -> None:
        """Force the episode to end by setting all agents to truncated state."""
        self.__c_sim.truncations()[:] = True

    @with_instance_timer("step", timer_attr="_timer")
    def step(self) -> None:
        """Execute one timestep of the environment dynamics.

        Actions must be set beforehand using agent(i).set_action() or by setting
        actions directly via _c_sim.actions()[i] = action_idx.
        """
        self._timer.stop("thread_idle")

        with self._timer("c_sim.step"):
            self.__c_sim.step()

        for handler in self._event_handlers:
            with self._timer(f"sim.on_step.{handler.__class__.__name__}"):
                handler.on_step()

        if self.is_done():
            self._timer.start("episode_end")
            for handler in self._event_handlers:
                with self._timer(f"sim.on_episode_end.{handler.__class__.__name__}"):
                    handler.on_episode_end()
            self._timer.stop("episode_end")

        self._timer.start("thread_idle")

    def close(self) -> None:
        """Close the environment."""
        for handler in self._event_handlers:
            handler.on_close()
        del self.__c_sim
        # Notify simulator that this simulation is closed
        if self._simulator is not None:
            self._simulator._on_simulation_closed(self)

    @property
    def config(self) -> mettagrid_config.MettaGridConfig:
        return self._config

    @property
    def _c_sim(self) -> MettaGridCpp:
        return self.__c_sim

    @property
    def episode_rewards(self) -> np.ndarray:
        """Get the episode rewards."""
        return self.__c_sim.get_episode_rewards()

    @property
    def episode_stats(self) -> EpisodeStats:
        return self.__c_sim.get_episode_stats()

    @property
    def current_step(self) -> int:
        return self.__c_sim.current_step

    @property
    def num_agents(self) -> int:
        return self._config.game.num_agents

    @property
    def action_ids(self) -> dict[str, int]:
        return self._action_ids

    @property
    def action_names(self) -> list[str]:
        return list(self._action_ids.keys())

    @property
    def object_type_names(self) -> list[str]:
        return self.__c_sim.object_type_names

    @property
    def resource_names(self) -> list[str]:
        return self._config.game.resource_names

    @property
    def features(self) -> Sequence[ObservationFeatureSpec]:
        return list(self._features.values())

    def get_feature(self, feature_id: int) -> ObservationFeatureSpec:
        """Get a feature by its ID."""
        return self._features[feature_id]

    @property
    def action_success(self) -> list[bool]:
        return self.__c_sim.action_success()

    @property
    def num_observation_tokens(self) -> int:
        return self.config.game.obs.num_tokens

    @property
    def observation_shape(self) -> tuple:
        return (self.num_observation_tokens, self.config.game.obs.token_dim)

    @property
    def map_width(self) -> int:
        """Get the width of the map."""
        return self.__c_sim.map_width

    @property
    def map_height(self) -> int:
        """Get the height of the map."""
        return self.__c_sim.map_height

    @property
    def seed(self) -> int:
        return self._seed

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
        return self.__c_sim.grid_objects(bbox.min_row, bbox.max_row, bbox.min_col, bbox.max_col, ignore_list)

    def _make_map(self) -> GameMap:
        map_builder = self._config.game.map_builder.create()
        game_map = map_builder.build_for_num_agents(self._config.game.num_agents)

        return game_map


class Simulator:
    def __init__(self):
        self._config_invariants = None
        self._event_handlers = []
        self._current_simulation = None

    def add_event_handler(self, handler: SimulatorEventHandler) -> None:
        self._event_handlers.append(handler)

    def new_simulation(
        self, config: mettagrid_config.MettaGridConfig, seed: int = 0, buffers: Optional[Buffers] = None
    ) -> Simulation:
        assert self._current_simulation is None, "A simulation is already running"
        if self._config_invariants is None:
            self._config_invariants = self._compute_config_invariants(config)

        config_invariants = self._compute_config_invariants(config)
        if self._config_invariants != config_invariants:
            logger.error("Config invariants have changed")
            logger.error(f"Old invariants: {self._config_invariants}")
            logger.error(f"New invariants: {config_invariants}")
            raise ValueError("Config invariants have changed")

        self._current_simulation = Simulation(
            config=config, seed=seed, event_handlers=self._event_handlers, simulator=self, buffers=buffers
        )
        return self._current_simulation

    def _on_simulation_closed(self, simulation: Simulation) -> None:
        """Called by Simulation.close() to notify that simulation has closed."""
        if self._current_simulation is simulation:
            self._current_simulation = None

    def close(self) -> None:
        """Shut down the simulator."""
        if self._current_simulation is not None:
            self._current_simulation.close()

    def _compute_config_invariants(self, config: mettagrid_config.MettaGridConfig) -> dict[str, Any]:
        return {
            "num_agents": config.game.num_agents,
            "action_names": [action.name for action in config.game.actions.actions()],
            "object_type_names": config.game.objects.keys(),
            "resource_names": config.game.resource_names,
            "vibe_names": config.game.vibe_names,
        }


class SimulationAgent:
    def __init__(self, sim: Simulation, agent_id: int):
        self._sim = sim
        self._agent_id = agent_id

    @property
    def id(self) -> int:
        return self._agent_id

    def set_action(self, action: Action | str) -> None:
        # Convert action to index
        action_name = action if isinstance(action, str) else action.name
        action_idx = self._sim.action_ids[action_name]
        self._sim._c_sim.actions()[self._agent_id] = action_idx

    @property
    def observation(self) -> AgentObservation:
        tokens = []
        agent_obs = self._sim._c_sim.observations()[self._agent_id]
        for o in agent_obs:
            (location, feature_id, value) = o
            if feature_id == 0xFF:
                break
            tokens.append(
                ObservationToken(
                    feature=self._sim.get_feature(feature_id),
                    location=PackedCoordinate.unpack(location) or (0, 0),
                    value=int(value),
                    raw_token=o,
                )
            )
        return AgentObservation(agent_id=self._agent_id, tokens=tokens)

    @property
    def step_reward(self) -> float:
        return self._sim._c_sim.rewards()[self._agent_id]

    @property
    def episode_reward(self) -> float:
        return self._sim._c_sim.get_episode_rewards()[self._agent_id]

    @property
    def last_action_success(self) -> bool:
        return self._sim._c_sim.action_success()[self._agent_id]

    @property
    def inventory(self) -> Dict[str, int]:
        """Get the agent's current inventory from observations.

        Inventory tokens appear at the agent's center position in the observation window.
        Returns a dictionary mapping resource names to their quantities.
        """
        inv = {}
        obs = self.observation

        for token in obs.tokens:
            # Check if this is an inventory feature
            if token.feature.name.startswith("inv:"):
                # Extract resource name from "inv:resource_name" format
                resource_name = token.feature.name[4:]  # Remove "inv:" prefix
                inv[resource_name] = token.value

        return inv

    @property
    def global_observations(self) -> Dict[str, int]:
        """Get global observation tokens from observations.

        Global observation tokens appear at the agent's center position in the observation window,
        along with agent-specific observations. This includes features like episode_completion_pct,
        last_action, and last_reward.

        Returns a dictionary mapping feature names to their values.
        """
        global_obs = {}
        obs = self.observation

        # Global observation feature names
        global_feature_names = {
            "episode_completion_pct",
            "last_action",
            "last_reward",
        }

        for token in obs.tokens:
            # Check if this is a global observation feature
            if token.feature.name in global_feature_names:
                global_obs[token.feature.name] = token.value

        return global_obs

    def set_inventory(self, inventory: Dict[str, int]) -> None:
        """Set an agent's inventory by resource name.

        Any resources not mentioned will be cleared in the underlying C++ call.
        """

        self._sim._c_sim.set_inventory(
            self._agent_id,
            {self._sim.resource_names.index(name): int(amount) for name, amount in inventory.items()},
        )
