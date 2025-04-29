from mettagrid.config.utils import get_test_basic_cfg

# Make sure all modules import without errors:
from mettagrid.mettagrid_env import MettaGridEnv

# Make sure all dependencies are installed:


def main():
    cfg = get_test_basic_cfg()
    # Create the environment:
    mettaGridEnv = MettaGridEnv(render_mode=None, env_cfg=cfg)

    # reset the environment a few times to make sure no memory is leaked:
    for _ in range(10):
        mettaGridEnv.reset()

    mettaGridEnv = None


if __name__ == "__main__":
    main()
