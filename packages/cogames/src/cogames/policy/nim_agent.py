"""FFI-backed scripted Nim policy for CoGames."""

from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

from cogames.policy.interfaces import AgentPolicy, Policy
from mettagrid import MettaGridAction, MettaGridEnv, MettaGridObservation, dtype_actions
from mettagrid.mettagrid_c import PackedCoordinate

# Keep this in sync with scripted_agent.nim
_RESOURCE_ORDER = ["carbon", "oxygen", "germanium", "silicon", "energy", "heart"]
_RESOURCE_COUNT = len(_RESOURCE_ORDER)
_FEATURE_TO_RESOURCE = {
    17: "energy",
    18: "carbon",
    19: "oxygen",
    20: "germanium",
    21: "silicon",
    22: "heart",
}
_RESOURCE_INDEX = {name: idx for idx, name in enumerate(_RESOURCE_ORDER)}
_EXTRACTOR_TYPES = {
    "carbon_extractor",
    "oxygen_extractor",
    "germanium_extractor",
    "silicon_extractor",
}


class _ObservedTileDto(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("terrain", ctypes.c_int32),
        ("station", ctypes.c_int32),
        ("cooldownEnds", ctypes.c_int32),
    ]


class _AgentViewDto(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int32),
        ("posX", ctypes.c_int32),
        ("posY", ctypes.c_int32),
        ("energy", ctypes.c_int32),
        ("cooldown", ctypes.c_int32),
        ("inventory", ctypes.c_int32 * _RESOURCE_COUNT),
        ("vibe", ctypes.c_int32),
        ("teammatesPtr", ctypes.POINTER(ctypes.c_int32)),
        ("teammateCount", ctypes.c_int32),
        ("sharedVibesPtr", ctypes.POINTER(ctypes.c_int32)),
        ("sharedVibesCount", ctypes.c_int32),
        ("observedPtr", ctypes.POINTER(_ObservedTileDto)),
        ("observedCount", ctypes.c_int32),
    ]


class _EnvViewDto(ctypes.Structure):
    _fields_ = [
        ("tick", ctypes.c_int32),
        ("assemblerCooldown", ctypes.c_int32),
        ("heartCost", ctypes.c_int32),
        ("targetHearts", ctypes.c_int32),
        ("teamHeartInventory", ctypes.c_int32),
    ]


class _AgentActionDto(ctypes.Structure):
    _fields_ = [
        ("moveDir", ctypes.c_int32),
        ("interactDir", ctypes.c_int32),
        ("emitVibe", ctypes.c_int32),
    ]


_MOVE_LOOKUP: Dict[int, str] = {
    0: "move_north",
    1: "move_south",
    2: "move_west",
    3: "move_east",
    4: "move_north",  # diagonals approximated to cardinals for now
    5: "move_north",
    6: "move_south",
    7: "move_south",
}

_VIBE_SIGNAL_TO_NAME: Dict[int, str] = {
    1: "carbon",
    2: "oxygen",
    3: "germanium",
    4: "silicon",
    5: "charger",
}


@dataclass
class _TileObservation:
    offset_x: int
    offset_y: int
    station: int
    terrain: int
    cooldown: int


@dataclass
class _ObservationSnapshot:
    energy: int
    inventory: Dict[str, int]
    cooldown: int
    vibe: int
    last_action: Optional[int]
    last_action_arg: Optional[int]
    tiles: list[_TileObservation]


def _default_library_path() -> Path:
    base = Path(__file__).resolve().parents[3]
    lib_dir = base / "nim"
    system = platform.system()
    if system == "Darwin":
        lib_name = "libcogames_agent.dylib"
    elif system == "Windows":
        lib_name = "libcogames_agent.dll"
    else:
        lib_name = "libcogames_agent.so"
    return lib_dir / lib_name


def _load_library(path: Optional[Path] = None) -> ctypes.CDLL:
    candidate = path
    override = os.getenv("COGAMES_NIM_AGENT_LIB")
    if candidate is None and override:
        candidate = Path(override)
    if candidate is None:
        candidate = _default_library_path()
    lib_path = candidate.expanduser()
    if not lib_path.exists():
        raise FileNotFoundError(f"Nim scripted agent library not found at {lib_path}")
    lib = ctypes.CDLL(str(lib_path))

    lib.cogames_agent_create.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib.cogames_agent_create.restype = ctypes.c_void_p

    lib.cogames_agent_reset.argtypes = []
    lib.cogames_agent_reset.restype = None

    lib.cogames_agent_register_buffers.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cogames_agent_register_buffers.restype = None

    lib.cogames_agent_step.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(_AgentViewDto),
        ctypes.POINTER(_EnvViewDto),
    ]
    lib.cogames_agent_step.restype = _AgentActionDto
    return lib


@dataclass
class _VibeMapping:
    name_to_index: Dict[str, int]
    signal_to_action: Dict[int, int]


class NimAgentController:
    """Thin wrapper around the Nim shared library controller."""

    def __init__(
        self,
        env: MettaGridEnv,
        seed: int | None = None,
        heart_cost_override: Optional[int] = None,
        library_path: Optional[Path] = None,
    ):
        self._lib = _load_library(library_path)
        mg_config = env.mg_config

        heart_cost = heart_cost_override
        assembler_cfg = mg_config.game.objects.get("assembler")
        if heart_cost is None and assembler_cfg is not None:
            heart_cost = getattr(assembler_cfg, "heart_cost", None)
        if heart_cost is None:
            heart_cost = 10

        self._controller_ptr = self._lib.cogames_agent_create(
            ctypes.c_int32(seed or 0), ctypes.c_int32(int(heart_cost))
        )
        self._tick = 0

        self._env = env
        self._action_lookup = {name: idx for idx, name in enumerate(env.action_names)}
        self._noop_index = self._action_lookup.get("noop", 0)
        self._move_deltas = self._build_move_deltas()

        vibe_names = list(mg_config.game.vibe_names or [])
        self._vibe_mapping = self._build_vibe_mapping(vibe_names)

        num_agents = mg_config.game.num_agents
        teammate_ids = list(range(num_agents))
        self._teammate_array = (ctypes.c_int32 * len(teammate_ids))(*teammate_ids)
        self._empty_shared_vibes = (ctypes.c_int32 * 0)()
        self._scratch_observed: Dict[int, ctypes.Array[_ObservedTileDto]] = {}

        self._agent_positions: Dict[int, tuple[int, int]] = {agent_id: (0, 0) for agent_id in teammate_ids}
        self._last_energy: Dict[int, Optional[int]] = {agent_id: None for agent_id in teammate_ids}
        self._last_sent_action: Dict[int, Optional[int]] = {agent_id: None for agent_id in teammate_ids}

        self._heart_cost = int(heart_cost)
        self._target_hearts = getattr(mg_config.game.agent.rewards, "target_hearts", 1)
        self._object_types = list(env.object_type_names)
        obs_width = getattr(mg_config.game, "obs_width", 11) or 11
        self._obs_center = obs_width // 2

    def reset(self) -> None:
        self._lib.cogames_agent_reset()
        self._tick = 0
        for agent_id in self._agent_positions:
            self._agent_positions[agent_id] = (0, 0)
            self._last_energy[agent_id] = None
            self._last_sent_action[agent_id] = None
            self._scratch_observed.pop(agent_id, None)

    def step(self, agent_id: int, obs: MettaGridObservation) -> MettaGridAction:
        snapshot = self._parse_observation(obs)
        self._apply_last_action_feedback(agent_id, snapshot)

        agent_view = _AgentViewDto()
        agent_view.id = ctypes.c_int32(agent_id)

        pos_x, pos_y = self._agent_positions[agent_id]
        agent_view.posX = pos_x
        agent_view.posY = pos_y
        agent_view.energy = snapshot.energy
        agent_view.cooldown = snapshot.cooldown

        for resource, idx in _RESOURCE_INDEX.items():
            agent_view.inventory[idx] = snapshot.inventory.get(resource, 0)

        agent_view.vibe = snapshot.vibe
        agent_view.teammatesPtr = ctypes.cast(self._teammate_array, ctypes.POINTER(ctypes.c_int32))
        agent_view.teammateCount = ctypes.c_int32(len(self._teammate_array))
        agent_view.sharedVibesPtr = ctypes.cast(self._empty_shared_vibes, ctypes.POINTER(ctypes.c_int32))
        agent_view.sharedVibesCount = ctypes.c_int32(0)

        observed_tiles = self._to_observed_tiles(agent_id, snapshot.tiles)
        if observed_tiles:
            observed_array = (_ObservedTileDto * len(observed_tiles))(*observed_tiles)
            self._scratch_observed[agent_id] = observed_array
            agent_view.observedPtr = ctypes.cast(observed_array, ctypes.POINTER(_ObservedTileDto))
            agent_view.observedCount = ctypes.c_int32(len(observed_tiles))
        else:
            self._scratch_observed.pop(agent_id, None)
            agent_view.observedPtr = ctypes.POINTER(_ObservedTileDto)()
            agent_view.observedCount = ctypes.c_int32(0)

        env_view = _EnvViewDto()
        env_view.tick = ctypes.c_int32(self._tick)
        env_view.assemblerCooldown = ctypes.c_int32(0)
        env_view.heartCost = ctypes.c_int32(self._heart_cost)
        env_view.targetHearts = ctypes.c_int32(self._target_hearts)
        env_view.teamHeartInventory = ctypes.c_int32(0)

        action = self._lib.cogames_agent_step(
            ctypes.c_int32(agent_id), ctypes.byref(agent_view), ctypes.byref(env_view)
        )
        action_idx = self._map_action(action)

        self._last_sent_action[agent_id] = action_idx
        self._last_energy[agent_id] = snapshot.energy
        self._tick += 1
        return dtype_actions.type(action_idx)

    def _map_action(self, action: _AgentActionDto) -> int:
        vibe_signal = int(action.emitVibe)
        if vibe_signal in self._vibe_mapping.signal_to_action:
            return self._vibe_mapping.signal_to_action[vibe_signal]

        move_dir = int(action.moveDir)
        move_name = _MOVE_LOOKUP.get(move_dir)
        if move_name:
            idx = self._action_lookup.get(move_name)
            if idx is not None:
                return idx

        return self._noop_index

    def _build_vibe_mapping(self, vibe_names: Sequence[str]) -> _VibeMapping:
        name_to_index = {name: idx for idx, name in enumerate(vibe_names)}
        signal_to_action: Dict[int, int] = {}
        for signal, vibe_name in _VIBE_SIGNAL_TO_NAME.items():
            vibe_idx = name_to_index.get(vibe_name)
            if vibe_idx is None:
                continue
            action_name = f"change_vibe_{vibe_idx}"
            action_idx = self._action_lookup.get(action_name)
            if action_idx is not None:
                signal_to_action[signal] = action_idx
        return _VibeMapping(name_to_index=name_to_index, signal_to_action=signal_to_action)

    def _build_move_deltas(self) -> Dict[int, tuple[int, int]]:
        deltas: Dict[int, tuple[int, int]] = {}
        for name, idx in self._action_lookup.items():
            if name == "move_north":
                deltas[idx] = (0, -1)
            elif name == "move_south":
                deltas[idx] = (0, 1)
            elif name == "move_west":
                deltas[idx] = (-1, 0)
            elif name == "move_east":
                deltas[idx] = (1, 0)
        return deltas

    def _parse_observation(self, obs: MettaGridObservation) -> _ObservationSnapshot:
        tokens = obs[0]
        inventory: Dict[str, int] = {}
        energy = 0
        cooldown = 0
        vibe = 0
        last_action: Optional[int] = None
        last_action_arg: Optional[int] = None
        tile_data: Dict[tuple[int, int], Dict[str, int]] = {}

        for packed, feature_id, value in tokens:
            coord = PackedCoordinate.unpack(int(packed))
            if coord is None:
                break
            row, col = coord
            fid = int(feature_id)
            val = int(value)

            if fid == 11:
                vibe = val
            elif fid == 8:
                last_action = val
            elif fid == 9:
                last_action_arg = val
            elif fid == 14:
                if row == self._obs_center and col == self._obs_center:
                    cooldown = val
                tile = tile_data.setdefault((row, col), {})
                tile["cooldown"] = val
            elif fid == 0:
                tile = tile_data.setdefault((row, col), {})
                tile["type_id"] = val
            elif fid in _FEATURE_TO_RESOURCE:
                resource = _FEATURE_TO_RESOURCE[fid]
                inventory[resource] = val
                if resource == "energy":
                    energy = val

        tiles: list[_TileObservation] = []
        for (row, col), data in tile_data.items():
            if row == self._obs_center and col == self._obs_center:
                continue
            type_id = data.get("type_id")
            station, terrain = self._classify_tile(type_id)
            if station == 0 and terrain == 1 and data.get("cooldown", 0) == 0:
                continue
            offset_x = col - self._obs_center
            offset_y = row - self._obs_center
            tiles.append(
                _TileObservation(
                    offset_x=offset_x,
                    offset_y=offset_y,
                    station=station,
                    terrain=terrain,
                    cooldown=data.get("cooldown", 0),
                )
            )

        return _ObservationSnapshot(
            energy=energy,
            inventory=inventory,
            cooldown=cooldown,
            vibe=vibe,
            last_action=last_action,
            last_action_arg=last_action_arg,
            tiles=tiles,
        )

    def _classify_tile(self, type_id: Optional[int]) -> tuple[int, int]:
        if type_id is None or type_id < 0 or type_id >= len(self._object_types):
            return 0, 1
        name = self._object_types[type_id] or ""
        if name == "wall":
            return 0, 2
        if name == "charger":
            return 1, 1
        if name in _EXTRACTOR_TYPES:
            return 2, 1
        if name == "assembler":
            return 4, 1
        if name == "chest":
            return 5, 1
        return 0, 1

    def _apply_last_action_feedback(self, agent_id: int, snapshot: _ObservationSnapshot) -> None:
        last_sent = self._last_sent_action.get(agent_id)
        if last_sent is None:
            return

        delta = self._move_deltas.get(last_sent)
        if delta is not None:
            success_list = getattr(self._env, "action_success", None)
            success = False
            if isinstance(success_list, list) and agent_id < len(success_list):
                success = bool(success_list[agent_id])
            if success:
                x, y = self._agent_positions.get(agent_id, (0, 0))
                dx, dy = delta
                self._agent_positions[agent_id] = (x + dx, y + dy)
        self._last_sent_action[agent_id] = None

    def _to_observed_tiles(self, agent_id: int, tiles: Sequence[_TileObservation]) -> list[_ObservedTileDto]:
        base_x, base_y = self._agent_positions.get(agent_id, (0, 0))
        observed: list[_ObservedTileDto] = []
        for tile in tiles:
            observed.append(
                _ObservedTileDto(
                    x=base_x + tile.offset_x,
                    y=base_y + tile.offset_y,
                    terrain=tile.terrain,
                    station=tile.station,
                    cooldownEnds=tile.cooldown,
                )
            )
        return observed


class NimAgentPolicyImpl(AgentPolicy):
    """Per-agent wrapper that routes observations through the Nim controller."""

    def __init__(self, controller: NimAgentController, agent_id: int):
        self._controller = controller
        self._agent_id = agent_id

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        return self._controller.step(self._agent_id, obs)

    def reset(self) -> None:
        self._controller.reset()


class NimScriptedPolicy(Policy):
    """Policy that defers action selection to the scripted Nim agent."""

    def __init__(
        self,
        env: MettaGridEnv,
        seed: int | None = None,
        heart_cost: int | None = None,
        library_path: Optional[Path] = None,
    ):
        if not isinstance(env, MettaGridEnv):
            raise TypeError("NimScriptedPolicy requires a MettaGridEnv instance")
        self._controller = NimAgentController(env, seed=seed, heart_cost_override=heart_cost, library_path=library_path)

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return NimAgentPolicyImpl(self._controller, agent_id)
