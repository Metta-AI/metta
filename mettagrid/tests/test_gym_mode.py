import random

import gymnasium as gym
import numpy as np
import pytest

from mettagrid.mettagrid_c import MettaGrid
from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.util.actions import generate_valid_random_actions
from mettagrid.util.hydra import get_cfg

# Define a constant seed for deterministic behavior
TEST_SEED = 42


# Rebuild the NumPy types using the exposed function
np_observations_type = np.dtype(MettaGrid.get_numpy_type_name("observations"))
np_terminals_type = np.dtype(MettaGrid.get_numpy_type_name("terminals"))
np_truncations_type = np.dtype(MettaGrid.get_numpy_type_name("truncations"))
np_rewards_type = np.dtype(MettaGrid.get_numpy_type_name("rewards"))
np_actions_type = np.dtype(MettaGrid.get_numpy_type_name("actions"))
np_masks_type = np.dtype(MettaGrid.get_numpy_type_name("masks"))
np_success_type = np.dtype(MettaGrid.get_numpy_type_name("success"))


@pytest.fixture
def environment():
    """Create and initialize the environment with a fixed seed."""
    # Set seeds for all random number generators
    np.random.seed(TEST_SEED)
    random.seed(TEST_SEED)
    cfg = get_cfg("benchmark")
    env = MettaGridEnv(cfg, render_mode="human", _recursive_=False, seed=TEST_SEED)
    return env


def test_mettagrid_env_gym_compat(environment: MettaGridEnv):
    # Basic Gym compliance
    assert isinstance(environment, gym.Env)
    assert isinstance(environment.observation_space, gym.Space)
    assert isinstance(environment.action_space, gym.Space)

    c_env: MettaGrid = environment._c_env
    assert c_env.is_gym_mode() is False

    c_env.reset()
    assert c_env.is_gym_mode() is True

    # Use safe, valid actions
    num_agents = environment.num_agents
    valid_actions = generate_valid_random_actions(environment, num_agents, seed=TEST_SEED)

    obs, reward, terminated, truncated, info = environment.step(valid_actions)

    assert environment.observation_space.contains(obs)

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np_observations_type

    assert isinstance(reward, np.ndarray)
    assert reward.dtype == np_rewards_type

    assert isinstance(terminated, np.ndarray)
    assert terminated.dtype == np_terminals_type

    assert isinstance(truncated, np.ndarray)
    assert truncated.dtype == np_truncations_type

    assert isinstance(info, dict)

    assert c_env.is_gym_mode() is True

    environment.close()
