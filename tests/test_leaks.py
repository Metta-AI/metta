import hydra
import numpy as np

# Make sure all modules import without errors:
from mettagrid.mettagrid_env import MettaGridEnv

# Make sure all dependencies are installed:
import hydra

@hydra.main(version_base=None, config_path="../configs", config_name="test_basic")
def main(cfg):
    # Create the environment:
    mettaGridEnv = MettaGridEnv(render_mode=None, **cfg)

    # reset the environment a few times to make sure no memory is leaked:
    for _ in range(10):
        mettaGridEnv.reset()
    
    mettaGridEnv = None


if __name__ == "__main__":
    main()
