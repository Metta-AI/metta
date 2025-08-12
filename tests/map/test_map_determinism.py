from __future__ import annotations

import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.test_support import TestEnvironmentBuilder


def _env_from_map(game_map: np.ndarray, num_agents: int, seed: int) -> MettaGrid:
    cfg = TestEnvironmentBuilder.make_test_config(num_agents=num_agents, map=game_map.tolist())
    map_data = cfg.pop("map")
    return MettaGrid(from_mettagrid_config(cfg), map_data, seed)


class TestInitialGridHashDeterminism:
    def test_same_map_same_hash_even_with_different_env_seed(self) -> None:
        game_map = TestEnvironmentBuilder.create_basic_grid(6, 5)
        game_map = TestEnvironmentBuilder.place_agents(game_map, [(2, 2), (2, 3)], agent_type="agent.player")

        env1 = _env_from_map(game_map, num_agents=2, seed=123)
        env2 = _env_from_map(game_map, num_agents=2, seed=456)

        assert env1.initial_grid_hash == env2.initial_grid_hash

    def test_different_maps_different_hash(self) -> None:
        base_map = TestEnvironmentBuilder.create_basic_grid(6, 5)
        map1 = TestEnvironmentBuilder.place_agents(base_map.copy(), [(2, 2), (2, 3)], agent_type="agent.player")
        map2 = TestEnvironmentBuilder.place_agents(base_map.copy(), [(2, 2), (3, 3)], agent_type="agent.player")

        env1 = _env_from_map(map1, num_agents=2, seed=123)
        env2 = _env_from_map(map2, num_agents=2, seed=123)

        assert env1.initial_grid_hash != env2.initial_grid_hash
