import hydra
import numpy as np

import mettagrid
import mettagrid.mettagrid_env


# This function will be recognized as a test by pytest
def test_dependencies():
    """Test that all required dependencies can be imported."""
    dependencies = [
        "hydra",
        "jmespath",
        "matplotlib",
        "pettingzoo",
        "pynvml",
        "pytest",
        "yaml",
        "raylib",
        "rich",
        "scipy",
        "tabulate",
        "tensordict",
        "torchrl",
        "termcolor",
        "wandb",
        "wandb_core",
        "pandas",
        "tqdm",
    ]

    missing_deps = []
    for dep in dependencies:
        try:
            # Use globals() to store the imported module, avoiding linter complaints
            globals()[dep] = __import__(dep)
            print(f"Successfully imported {dep}")
        except ImportError as e:
            missing_deps.append(f"{dep}: {str(e)}")

    if missing_deps:
        print("Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        raise ImportError("Missing required dependencies")

    # Check mettagrid modules
    mettagrid_modules = [
        "mettagrid.objects",
        "mettagrid.observation_encoder",
        "mettagrid.actions.attack",
        "mettagrid.actions.move",
        "mettagrid.actions.noop",
        "mettagrid.actions.rotate",
        "mettagrid.actions.swap",
        "mettagrid.action_handler",
        "mettagrid.event",
        "mettagrid.grid_env",
        "mettagrid.grid_object",
    ]

    for module_name in mettagrid_modules:
        try:
            __import__(module_name)
            print(f"Successfully imported {module_name}")
        except ImportError as err:
            raise ImportError(f"Failed to import {module_name}: {str(err)}") from err


def test_env_functionality():
    """Test basic environment functionality with hydra config."""
    config_path = "../configs"
    config_name = "test_basic"

    with hydra.initialize(version_base=None, config_path=config_path):
        cfg = hydra.compose(config_name=config_name)

        # Create the environment:
        mettaGridEnv = mettagrid.mettagrid_env.MettaGridEnv(cfg, render_mode=None)

        # Make sure the environment was created correctly:
        assert mettaGridEnv._c_env is not None
        assert mettaGridEnv._grid_env is not None
        assert mettaGridEnv._c_env == mettaGridEnv._grid_env
        assert mettaGridEnv.done is False

        # Make sure reset works:
        mettaGridEnv.reset()

        # Run a single step:
        assert mettaGridEnv._c_env.current_timestep() == 0
        (obs, rewards, terminated, truncated, infos) = mettaGridEnv.step([[0, 0]] * 5)
        assert mettaGridEnv._c_env.current_timestep() == 1

        # We have 5 agents, ~22 channels, 11x11 grid
        [num_agents, grid_width, grid_height, num_channels] = obs.shape
        assert num_agents == 5
        assert grid_width == 11
        assert grid_height == 11
        assert 20 <= num_channels <= 50
        assert rewards.shape == (5,)
        assert np.array_equal(terminated, [0, 0, 0, 0, 0])
        assert np.array_equal(truncated, [0, 0, 0, 0, 0])

        # Test finalize_episode
        mettaGridEnv.finalize_episode()

        # Test environment properties
        assert mettaGridEnv._max_steps == 5000
        assert mettaGridEnv.single_observation_space.shape == (grid_width, grid_height, num_channels)
        [num_actions, max_arg] = mettaGridEnv.single_action_space.nvec.tolist()
        assert 5 <= num_actions <= 20, f"num_actions: {num_actions}"
        assert 5 <= max_arg <= 20, f"max_arg: {max_arg}"
        assert mettaGridEnv.render_mode is None
        assert mettaGridEnv._c_env.map_width() == 25
        assert mettaGridEnv._c_env.map_height() == 25
        assert mettaGridEnv._c_env.num_agents() == 5
        assert mettaGridEnv.action_success.shape == (5,)
        assert mettaGridEnv.object_type_names() == mettaGridEnv._c_env.object_type_names()
