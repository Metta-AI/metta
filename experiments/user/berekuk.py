import random

import mettagrid
import mettagrid.mapgen.mapgen
import mettagrid.mapgen.scene
import mettagrid.mapgen.scenes.inline_ascii


def mapgen_transform_demo() -> mettagrid.MettaGridConfig:
    mg_config = mettagrid.MettaGridConfig()
    mg_config.game.num_agents = 0
    mg_config.game.map_builder = mettagrid.mapgen.mapgen.MapGen.Config(
        num_agents=mg_config.game.num_agents,
        border_width=1,
        instance=mettagrid.mapgen.scenes.inline_ascii.InlineAscii.Config(
            data="""
.....
.#...
.###.
.....
            """,
            transform=random.choice(list(mettagrid.mapgen.scene.GridTransform)),
        ),
    )
    return mg_config
