import numpy as np
import pytest

from mettagrid.config.utils import get_cfg
from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.resolvers import register_resolvers


@pytest.fixture
def environment():
    """Create and initialize the environment."""
    register_resolvers()
    cfg = get_cfg("benchmark")
    env = MettaGridEnv(cfg, render_mode="human", _recursive_=False)
    env.reset()
    yield env
    # Cleanup after test
    del env


def test_basic(environment):
    """
    Comprehensive test of MettaGrid environment functionality.
    This test combines the functionality of multiple tests into one
    while fixing the dtype issue for actions.
    """
    # ---- Test environment initialization ----
    assert environment._renderer is None
    assert environment._c_env is not None
    assert environment._grid_env is not None
    assert environment._c_env == environment._grid_env
    assert environment.done is False

    # ---- Test environment properties ----
    assert environment._max_steps > 0

    # Check observation space
    obs_shape = environment.single_observation_space.shape
    assert len(obs_shape) == 3  # (width, height, channels)
    assert obs_shape[0] > 0  # grid width
    assert obs_shape[1] > 0  # grid height
    assert obs_shape[2] > 0  # channels

    # Check action space
    [num_actions, max_arg] = environment.single_action_space.nvec.tolist()
    assert num_actions > 0, f"num_actions: {num_actions}"
    assert max_arg > 0, f"max_arg: {max_arg}"

    # Check env properties
    assert environment.render_mode == "human"
    assert environment._c_env.map_width() > 0
    assert environment._c_env.map_height() > 0
    num_agents = environment._c_env.num_agents()
    assert num_agents > 0
    assert environment.action_success.shape == (num_agents,)

    # Test object type names
    assert environment.object_type_names() == environment._c_env.object_type_names()

    # Test inventory item names
    assert environment.inventory_item_names() == environment._c_env.inventory_item_names()

    # Test action names
    assert environment.action_names() == environment._c_env.action_names()

    # ---- Test environment reset ----
    obs, info = environment.reset()

    # Check observation structure
    [agents_in_obs, grid_width, grid_height, num_channels] = obs.shape
    num_expected_agents = environment._c_env.num_agents()
    assert agents_in_obs == num_expected_agents
    assert grid_width > 0
    assert grid_height > 0
    assert 20 <= num_channels <= 50

    # ---- Test environment step ----
    # Check initial timestep
    assert environment._c_env.current_timestep() == 0

    # Take a step with NoOp actions for all agents
    # Create properly formatted actions with correct dtype
    actions = np.array([[0, 0]] * num_agents, dtype=np.int32)

    obs, rewards, terminated, truncated, infos = environment.step(actions)

    # Check timestep increased
    assert environment._c_env.current_timestep() == 1

    # Verify observation structure
    [agents_in_obs, grid_width, grid_height, num_channels] = obs.shape
    assert agents_in_obs == num_agents
    assert grid_width > 0
    assert grid_height > 0
    assert 20 <= num_channels <= 50

    # Verify rewards and termination flags
    assert rewards.shape == (num_agents,)
    assert len(terminated) == num_agents
    assert len(truncated) == num_agents

    # ---- Test episode stats ----
    infos = {}
    environment.process_episode_stats(infos)

    # ---- Additional testing with random actions ----
    # Reset for additional testing
    obs, info = environment.reset()

    # Test multiple steps with random actions
    for _i in range(5):
        # Generate random actions but ensure correct dtype
        random_actions = np.random.randint(low=0, high=[num_actions, max_arg], size=(num_agents, 2), dtype=np.int32)

        obs, rewards, terminated, truncated, infos = environment.step(random_actions)

        # Process episode stats if needed
        if np.any(terminated) or np.any(truncated):
            environment.process_episode_stats(infos)
            obs, info = environment.reset()

    # Final verification that environment is still functioning
    assert environment._c_env is not None
    assert environment._grid_env is not None
