from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np

# Don't use `from ... import ...` for convert_to_cpp_game_config or
# MettaGridConfig because it will cause a circular import.
import mettagrid.config.mettagrid_c_config as mettagrid_c_config
import mettagrid.config.mettagrid_config as mettagrid_config
from mettagrid.config.id_map import ObservationFeatureSpec
from mettagrid.map_builder.map_builder import GameMap
from mettagrid.mettagrid_c import MettaGrid as MettaGridCpp
from mettagrid.mettagrid_c import PackedCoordinate
from mettagrid.profiling.stopwatch import Stopwatch, with_instance_timer
from mettagrid.simulator.interface import AgentObservation, ObservationToken, SimulatorEventHandler
from mettagrid.simulator.map_cache import SharedMapCache, get_shared_cache
from mettagrid.simulator.types import Action

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
        event_handlers: Sequence[SimulatorEventHandler] = (),  # Use tuple to avoid mutable default
        simulator: Optional[Simulator] | None = None,
        buffers: Optional[Buffers] = None,
        maps_cache: Optional[SharedMapCache] = None,
    ):
        self._config = config
        self._seed = seed
        self._event_handlers = list(event_handlers)
        self._simulator = simulator
        self._maps_cache = maps_cache
        self._context: Dict[str, Any] = {}

        for handler in self._event_handlers:
            handler.set_simulation(self)

        self._timer = Stopwatch(log_level=logger.getEffectiveLevel())
        self._timer.start()

        game_config_dict = self._config.game.model_dump()

        with self._timer("sim.init.make_map"):
            map_grid = self._make_map().grid.tolist()

        # Create C++ config
        try:
            c_cfg = mettagrid_c_config.convert_to_cpp_game_config(game_config_dict)
        except Exception:
            logger.exception("Error creating C++ config")
            logger.error("Game config: %s", game_config_dict)
            raise

        # Create C++ environment
        with self._timer("sim.init.create_c_sim"):
            self.__c_sim = MettaGridCpp(c_cfg, map_grid, self._seed)

        # Compute action_ids from config actions
        self._action_ids: dict[str, int] = {
            action.name: idx for idx, action in enumerate(self._config.game.actions.actions())
        }

        # Set buffers on C++ simulation if provided (for PufferEnv shared memory)
        if buffers is not None:
            self.__c_sim.set_buffers(
                buffers.observations,
                buffers.terminals,
                buffers.truncations,
                buffers.rewards,
                buffers.actions,
            )

        # Build feature dict from id_map
        self._features: dict[int, ObservationFeatureSpec] = {
            feature.id: feature for feature in self._config.game.id_map().features()
        }

        self._start_episode()

        self._timer.start("sim.thread_idle")

    def agents(self) -> list[SimulationAgent]:
        return [self.agent(agent_id) for agent_id in range(self.num_agents)]

    def agent(self, agent_id: int) -> SimulationAgent:
        return SimulationAgent(self, agent_id)

    def observations(self) -> list[AgentObservation]:
        return [self.agent(agent_id).observation for agent_id in range(self.num_agents)]

    def is_done(self) -> bool:
        return bool(self.__c_sim.truncations().all() or self.__c_sim.terminals().all())

    @with_instance_timer("sim.episode.start", timer_attr="_timer")
    def _start_episode(self) -> None:
        """Start a new episode (internal use only)."""
        self._episode_started = True
        self._context = {}

        for handler in self._event_handlers:
            with self._timer(f"sim.on_episode_start.{handler.__class__.__name__}"):
                handler.on_episode_start()

    @with_instance_timer("sim.episode.end", timer_attr="_timer")
    def end_episode(self) -> None:
        """Force the episode to end by setting all agents to truncated state."""
        self.__c_sim.truncations()[:] = True

    @with_instance_timer("sim.step", timer_attr="_timer")
    def step(self) -> None:
        """Execute one timestep of the environment dynamics.

        Actions must be set beforehand using agent(i).set_action() or by setting
        actions directly via _c_sim.actions()[i] = action_idx.
        """
        self._timer.stop("sim.thread_idle")

        with self._timer("sim.step.c_sim"):
            self.__c_sim.step()

        for handler in self._event_handlers:
            with self._timer(f"sim.step.{handler.__class__.__name__.lower()}"):
                handler.on_step()

        if self.is_done():
            with self._timer("sim.episode.end"):
                for handler in self._event_handlers:
                    with self._timer(f"sim.episode.end.{handler.__class__.__name__.lower()}"):
                        handler.on_episode_end()

        self._timer.start("sim.thread_idle")

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
        if self._maps_cache is None:
            return self._config.game.map_builder.create().build_for_num_agents(self._config.game.num_agents)
        return self._maps_cache.get_or_create(self._config.game.map_builder, self._config.game.num_agents)


class Simulator:
    def __init__(self, maps_cache_size: Optional[int] = None):
        self._maps_cache = None
        if maps_cache_size is not None:
            self._maps_cache = get_shared_cache(maps_per_key=maps_cache_size)

        self._config_invariants = None
        self._event_handlers = []
        self._current_simulation = None

    def add_event_handler(self, handler: SimulatorEventHandler) -> None:
        self._event_handlers.append(handler)

    def new_simulation(
        self, config: mettagrid_config.MettaGridConfig, seed: int = 0, buffers: Optional[Buffers] = None
    ) -> Simulation:
        # Initialize invariants on first simulation
        if self._config_invariants is None:
            self._config_invariants = self._compute_config_invariants(config)

        # Check if invariants have changed
        config_invariants = self._compute_config_invariants(config)
        if self._config_invariants != config_invariants:
            # Allow updates between episodes for curriculum training
            self._config_invariants = config_invariants

        self._current_simulation = Simulation(
            config=config,
            seed=seed,
            event_handlers=self._event_handlers,
            simulator=self,
            buffers=buffers,
            maps_cache=self._maps_cache,
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

    def self_observation(self) -> list[ObservationToken]:
        """Get observation tokens for the agent itself.

        These are tokens at the agent's center position in the observation window,
        including inventory, global observations, and other agent-specific features.
        """
        obs = self.observation

        # Agent's own tokens are at the center of the observation window
        c_sim = self._sim._c_sim
        center = (c_sim.obs_height // 2, c_sim.obs_width // 2)

        return [token for token in obs.tokens if token.location == center]

    @property
    def inventory(self) -> Dict[str, int]:
        """Get the agent's current inventory from observations.

        Inventory tokens appear at the agent's center position in the observation window.
        Returns a dictionary mapping resource names to their quantities.

        Inventory values are encoded using multi-token encoding:
        - inv:{resource} contains the base value (amount % token_value_base)
        - inv:{resource}:p1 contains power 1 ((amount / token_value_base) % token_value_base)
        - inv:{resource}:p2 contains power 2 ((amount / token_value_base^2) % token_value_base)
        - etc.
        The full value is reconstructed as: base + p1 * B + p2 * B^2 + ... where B = token_value_base
        """
        import re

        token_value_base = self._sim._config.game.obs.token_value_base

        # Collect tokens by resource name and power
        inv_values: Dict[str, Dict[int, int]] = {}  # resource_name -> {power -> value}

        for token in self.self_observation():
            if token.feature.name.startswith("inv:"):
                suffix = token.feature.name[4:]  # Remove "inv:" prefix

                # Parse power suffix :pN where N is the power number
                match = re.match(r"^(.+):p(\d+)$", suffix)
                if match:
                    resource_name = match.group(1)
                    power = int(match.group(2))
                else:
                    resource_name = suffix
                    power = 0

                if resource_name not in inv_values:
                    inv_values[resource_name] = {}
                inv_values[resource_name][power] = token.value

        # Reconstruct full values from base and power tokens
        inv = {}
        for resource_name, power_values in inv_values.items():
            total = 0
            for power, value in power_values.items():
                total += value * (token_value_base**power)
            inv[resource_name] = total

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
