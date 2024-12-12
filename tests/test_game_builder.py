import hydra
import numpy as np
import mettagrid
import mettagrid.mettagrid_env

@hydra.main(version_base=None, config_path="../configs", config_name="test_basic")
def main(cfg):

    print("Basic level:")
    a = mettagrid.mettagrid_env.MettaGridEnv(render_mode=None, **cfg)
    print(a._c_env.render())

    print("Level with 2x2 rooms:")
    cfg.game.map.layout.rooms_x = 2
    cfg.game.map.layout.rooms_y = 2
    cfg.game.num_agents = 20
    b = mettagrid.mettagrid_env.MettaGridEnv(render_mode=None, **cfg)
    print(b._c_env.render())

    print("Level with 2x2 rooms without border,")
    print("but have a single map border:")
    cfg.game.map.layout.rooms_x = 2
    cfg.game.map.layout.rooms_y = 2
    cfg.game.num_agents = 20 # Must match 5 agents per room and 4 rooms.
    cfg.game.map.border = 1
    cfg.game.map.room.border = 0
    c = mettagrid.mettagrid_env.MettaGridEnv(render_mode=None, **cfg)
    print(c._c_env.render())

if __name__ == "__main__":
    main()
