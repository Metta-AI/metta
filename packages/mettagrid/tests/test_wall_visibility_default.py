from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.utils import create_grid
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME
from mettagrid.simulator import Simulation
from mettagrid.test_support import ObservationHelper


def test_walls_visible_without_tags():
    """Walls without tags are observable."""
    game_map = create_grid(5, 5, fill_value=".")
    game_map[0, :] = "#"
    game_map[-1, :] = "#"
    game_map[:, 0] = "#"
    game_map[:, -1] = "#"
    game_map[1, 1] = "@"

    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=50),
            max_steps=1,
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            objects={"wall": WallConfig()},
            map_builder=AsciiMapBuilder.Config(map_data=game_map.tolist(), char_to_map_name=DEFAULT_CHAR_TO_NAME),
        )
    )

    sim = Simulation(cfg)
    obs = sim._c_sim.observations()

    helper = ObservationHelper()
    tag_feature_id = sim.config.game.id_map().feature_id("tag")
    wall_tag_tokens = helper.find_tokens(obs[0], feature_id=tag_feature_id)
    assert len(wall_tag_tokens) > 0, "Walls should be visible even without tags"
