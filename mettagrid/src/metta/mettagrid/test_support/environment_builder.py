from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid, PackedCoordinate, dtype_actions  # noqa: F401
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


@dataclass
class EnvConfig:
    NUM_AGENTS: int = 2
    OBS_HEIGHT: int = 3
    OBS_WIDTH: int = 3
    NUM_OBS_TOKENS: int = 100
    OBS_TOKEN_SIZE: int = 3
    EMPTY_TOKEN = [0xFF, 0xFF, 0xFF]


class TestEnvironmentBuilder:
    """Helper class to build test environments with different configurations."""

    @staticmethod
    def create_basic_grid(width: int = 8, height: int = 4) -> np.ndarray:
        """Create a basic grid with walls around perimeter."""
        game_map = np.full((height, width), "empty", dtype="<U50")
        game_map[0, :] = "wall"
        game_map[-1, :] = "wall"
        game_map[:, 0] = "wall"
        game_map[:, -1] = "wall"
        return game_map

    @staticmethod
    def place_agents(game_map: np.ndarray, positions: List[Tuple[int, int]]) -> np.ndarray:
        """Place agents at specified positions.

        Note: positions are (y, x) indexing into the numpy array.
        """
        for _, (y, x) in enumerate(positions):
            game_map[y, x] = "agent.red"
        return game_map

    @staticmethod
    def create_environment(game_map: np.ndarray, max_steps: int = 10, num_agents: int | None = None) -> MettaGrid:
        """Create a MettaGrid environment from a game map."""
        if num_agents is None:
            num_agents = EnvConfig.NUM_AGENTS

        game_config = {
            "max_steps": max_steps,
            "num_agents": num_agents,
            "obs_width": EnvConfig.OBS_WIDTH,
            "obs_height": EnvConfig.OBS_HEIGHT,
            "num_observation_tokens": EnvConfig.NUM_OBS_TOKENS,
            "inventory_item_names": ["laser", "armor"],
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "attack": {"enabled": False},
                "put_items": {"enabled": False},
                "get_items": {"enabled": False},
                "swap": {"enabled": False},
                "change_color": {"enabled": False},
                "change_glyph": {"enabled": False, "number_of_glyphs": 4},
            },
            "groups": {"red": {"id": 0, "props": {}}},
            "objects": {"wall": {"type_id": 1}},
            "agent": {},
        }
        return MettaGrid(from_mettagrid_config(game_config), game_map.tolist(), 42)
