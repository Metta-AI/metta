# Replay loader for MettaGrid replays.
#
# Loads .json.z replay files and provides a Simulation-like interface for playback.
# Mirrors the nim replays.nim approach: explicit time-series field expansion.
#
# Version support:
# - V2: Supported directly (time-series format [[step, value], ...])
# - V3: Supported directly (same format, just compressed inventory)
# - V1: Not supported (requires conversion, done by nim/MettaScope)

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Supported replay format versions
# V2 and V3 use the same time-series format [[step, value], ...]
# V3 just has compressed inventory (but V2 test files already have this)
SUPPORTED_VERSIONS = frozenset({2, 3})

# Fields that use time-series format [[step, value], ...]
# These are expanded per-step. All other fields are constant.
# Matches nim replays.nim expand[] calls.
TIME_SERIES_FIELDS = frozenset(
    {
        "location",
        "orientation",
        "inventory",
        "color",
        "is_frozen",
        "frozen",  # alias
        "action_id",
        "action_parameter",
        "action_param",  # alias
        "action_success",
        "current_reward",
        "total_reward",
        "frozen_progress",
        "vibe_id",
        "production_progress",
        "cooldown_progress",
        "cooldown_remaining",
        "is_clipped",
        "is_clip_immune",
        "uses_count",
        "exhaustion",
        "cooldown_multiplier",
    }
)


def _expand_time_series(data: Any, step: int, default: Any = None) -> Any:
    # Expand a time-series field [[step, value], ...] to get value at step.
    # Matches nim replays.nim expand[] function.
    if data is None:
        return default
    if not isinstance(data, list):
        # Single constant value
        return data
    if len(data) == 0:
        return default
    # Check if it's time-series format: [[int, val], [int, val], ...]
    first = data[0]
    if not isinstance(first, (list, tuple)) or len(first) != 2:
        # Not time-series, it's a single array value (e.g., location [x, y])
        return data
    if not isinstance(first[0], int):
        # Not time-series (first element not an int step)
        return data
    # It's time-series format - find value at step
    result = default
    for update_step, value in data:
        if not isinstance(update_step, int):
            # Malformed, return data as-is
            return data
        if update_step <= step:
            result = value
        else:
            break
    return result


def _pairs_to_dict(pairs: Any) -> Dict[int, int]:
    # Convert [[itemId, count], ...] to {itemId: count, ...}.
    # Used for inventory and similar fields where components expect dict format.
    if not isinstance(pairs, list):
        return {}
    result: Dict[int, int] = {}
    for item in pairs:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            key, value = item
            if isinstance(key, int):
                result[key] = value
    return result


@dataclass
class ReplayData:
    # Replay metadata and configuration.
    version: int
    action_names: List[str]
    item_names: List[str]
    type_names: List[str]
    map_size: List[int]  # [width, height]
    num_agents: int
    max_steps: int
    mg_config: Dict[str, Any]
    objects: List[Dict[str, Any]]

    @staticmethod
    def load(path: Path | str) -> ReplayData:
        # Load and decompress the replay file.
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Replay file not found: {path}")

        with open(path, "rb") as f:
            compressed = f.read()

        try:
            decompressed = zlib.decompress(compressed)
            data = json.loads(decompressed.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to decompress/parse replay file: {e}") from e

        # Validate version
        version = data.get("version", 1)
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported replay version {version}. Supported versions: {sorted(SUPPORTED_VERSIONS)}")

        # Validate and extract map_size
        map_size = data.get("map_size", [0, 0])
        if not isinstance(map_size, list) or len(map_size) != 2:
            raise ValueError(f"Invalid map_size format: {map_size}. Expected [width, height].")
        if not all(isinstance(dim, int) and dim > 0 for dim in map_size):
            raise ValueError(f"Invalid map dimensions: {map_size}. Must be positive integers.")

        # Validate max_steps
        max_steps = data.get("max_steps", 0)
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError(f"Invalid max_steps: {max_steps}. Must be a positive integer.")

        # Validate num_agents
        num_agents = data.get("num_agents", 0)
        if not isinstance(num_agents, int) or num_agents < 0:
            raise ValueError(f"Invalid num_agents: {num_agents}. Must be a non-negative integer.")

        return ReplayData(
            version=version,
            action_names=data.get("action_names", []),
            item_names=data.get("item_names", []),
            type_names=data.get("type_names", []),
            map_size=map_size,
            num_agents=num_agents,
            max_steps=max_steps,
            mg_config=data.get("mg_config", {}),
            objects=data.get("objects", []),
        )

    @property
    def map_width(self) -> int:
        return self.map_size[0]

    @property
    def map_height(self) -> int:
        return self.map_size[1]


@dataclass
class ReplayConfig:
    # Minimal config structure needed by replay renderer.
    # Mimics MettaGridConfig.game structure for symbol rendering.
    game: "ReplayGameConfig"


@dataclass
class ReplayGameConfig:
    # Game config needed for rendering.
    objects: Dict[str, "ReplayObjectConfig"]
    resource_names: List[str]
    max_steps: int = 10000

    @staticmethod
    def from_mg_config(mg_config: Dict[str, Any]) -> ReplayGameConfig:
        game_cfg = mg_config.get("game", {})
        objects = {}
        for obj_name, obj_cfg in game_cfg.get("objects", {}).items():
            objects[obj_name] = ReplayObjectConfig(
                name=obj_cfg.get("name", obj_name),
                render_name=obj_cfg.get("render_name", obj_name),
                render_symbol=obj_cfg.get("render_symbol", "?"),
            )
        return ReplayGameConfig(
            objects=objects,
            resource_names=game_cfg.get("resource_names", []),
            max_steps=game_cfg.get("max_steps", 10000),
        )


@dataclass
class ReplayObjectConfig:
    name: str
    render_name: str
    render_symbol: str


@dataclass
class ReplaySimulation:
    # Simulation-like interface for replay playback.
    # Provides the interface needed by MiniscopeRenderer components.
    _replay: ReplayData
    _current_step: int = 0
    _config: Optional[ReplayConfig] = field(default=None, init=False)
    _object_states: Dict[int, Dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        # Build config from replay data
        game_config = ReplayGameConfig.from_mg_config(self._replay.mg_config)
        self._config = ReplayConfig(game=game_config)
        # Pre-compute object states at step 0
        self._rebuild_object_states()

    def _rebuild_object_states(self) -> None:
        # Rebuild object states for current step from replay data.
        self._object_states = {}
        for obj_idx, obj in enumerate(self._replay.objects):
            obj_state = self._get_object_state_at_step(obj, self._current_step)
            if obj_state:
                obj_id = obj_state.get("id", obj_idx)
                self._object_states[obj_id] = obj_state

    def _get_object_state_at_step(self, obj: Dict[str, Any], step: int) -> Dict[str, Any]:
        # Get object state at a specific step.
        # Matches nim replays.nim approach: explicit field handling.
        state: Dict[str, Any] = {}
        for key, value in obj.items():
            if key in TIME_SERIES_FIELDS:
                # Expand time-series field
                expanded = _expand_time_series(value, step)
                if expanded is not None:
                    state[key] = expanded
            else:
                # Constant field - use as-is
                state[key] = value

        # Convert location [col, row] to r, c keys for component compatibility
        if "location" in state:
            loc = state["location"]
            if isinstance(loc, (list, tuple)) and len(loc) == 2:
                state["c"] = loc[0]  # col
                state["r"] = loc[1]  # row

        # Convert inventory [[itemId, count], ...] to {itemId: count, ...}
        # Components expect dict format matching live simulation
        if "inventory" in state:
            state["inventory"] = _pairs_to_dict(state["inventory"])

        return state

    def set_step(self, step: int) -> None:
        # Set the current playback step.
        if step < 0:
            step = 0
        if step >= self._replay.max_steps:
            step = self._replay.max_steps - 1
        if step != self._current_step:
            self._current_step = step
            self._rebuild_object_states()

    def step_forward(self, count: int = 1) -> None:
        self.set_step(self._current_step + count)

    def step_backward(self, count: int = 1) -> None:
        self.set_step(self._current_step - count)

    # Properties matching Simulation interface

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def num_agents(self) -> int:
        return self._replay.num_agents

    @property
    def map_width(self) -> int:
        return self._replay.map_width

    @property
    def map_height(self) -> int:
        return self._replay.map_height

    @property
    def resource_names(self) -> List[str]:
        return self._replay.item_names

    @property
    def action_names(self) -> List[str]:
        return self._replay.action_names

    @property
    def object_type_names(self) -> List[str]:
        return self._replay.type_names

    @property
    def config(self) -> ReplayConfig:
        assert self._config is not None
        return self._config

    @property
    def max_steps(self) -> int:
        return self._replay.max_steps

    @property
    def episode_rewards(self) -> List[float]:
        # Get episode rewards from agent objects.
        rewards = [0.0] * self._replay.num_agents
        for obj_state in self._object_states.values():
            if obj_state.get("type_name") == "agent":
                agent_id = obj_state.get("agent_id")
                if agent_id is not None and 0 <= agent_id < len(rewards):
                    rewards[agent_id] = float(obj_state.get("total_reward", 0.0))
        return rewards

    def grid_objects(
        self, bbox: Optional[Any] = None, ignore_types: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, Any]]:
        # Return object states, optionally filtered by type.
        if ignore_types is None:
            return dict(self._object_states)
        return {
            obj_id: obj_state
            for obj_id, obj_state in self._object_states.items()
            if obj_state.get("type_name") not in ignore_types
        }

    def is_done(self) -> bool:
        return self._current_step >= self._replay.max_steps - 1


def load_replay(path: Path | str) -> ReplaySimulation:
    # Convenience function to load a replay and create a simulation.
    replay_data = ReplayData.load(path)
    return ReplaySimulation(_replay=replay_data)
