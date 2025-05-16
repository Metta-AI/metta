import random

import numpy as np
import pytest

from mettagrid.mettagrid_c import MettaGrid
from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.tests.utils import generate_valid_random_actions
from mettagrid.util.hydra import get_cfg

# Rebuild the NumPy types using the exposed function
np_observations_type = np.dtype(MettaGrid.get_numpy_type_name("observations"))
np_terminals_type = np.dtype(MettaGrid.get_numpy_type_name("terminals"))
np_truncations_type = np.dtype(MettaGrid.get_numpy_type_name("truncations"))
np_rewards_type = np.dtype(MettaGrid.get_numpy_type_name("rewards"))
np_actions_type = np.dtype(MettaGrid.get_numpy_type_name("actions"))
np_masks_type = np.dtype(MettaGrid.get_numpy_type_name("masks"))
np_success_type = np.dtype(MettaGrid.get_numpy_type_name("success"))

# Define a constant seed for deterministic behavior
TEST_SEED = 42


@pytest.fixture
def environment():
    """Create and initialize the environment with a fixed seed."""
    # Set seeds for all random number generators
    np.random.seed(TEST_SEED)
    random.seed(TEST_SEED)

    cfg = get_cfg("benchmark")
    env = MettaGridEnv(cfg, render_mode="human", _recursive_=False, seed=TEST_SEED)
    env.reset(seed=TEST_SEED)
    yield env
    # Cleanup after test
    del env


def test_basic(environment):
    """
    Comprehensive test of MettaGrid environment functionality.
    This test combines the functionality of multiple tests into one
    and ensures all actions are valid with deterministic behavior.
    """
    # Set seed again at the start of the test for consistent action generation
    np.random.seed(TEST_SEED)
    random.seed(TEST_SEED)

    # ---- Test environment initialization ----
    assert environment._renderer is None
    assert environment._c_env is not None
    assert environment.done is False

    # ---- Test environment properties ----
    assert environment.max_steps > 0

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
    assert environment.map_width > 0
    assert environment.map_height > 0
    num_agents = environment.num_agents
    assert num_agents > 0
    assert len(environment.action_success) == num_agents

    # Test object type names
    assert environment.object_type_names == environment._c_env.object_type_names()

    # Test inventory item names
    assert environment.inventory_item_names == environment._c_env.inventory_item_names()

    # Test action names
    assert environment.action_names == environment._c_env.action_names()

    # Test grid objects
    assert environment.grid_objects == environment._c_env.grid_objects()

    # ---- Test environment reset ----
    obs, info = environment.reset(seed=TEST_SEED)

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
    # Use our utility to generate valid actions for all agents
    actions = generate_valid_random_actions(
        environment,
        num_agents,
        force_action_type=0,  # First action type (likely NoOp or similar)
        force_action_arg=0,  # Argument 0 is valid for all action types
        seed=TEST_SEED,
    )

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
    # Reset for additional testing with seed
    obs, _info = environment.reset(seed=TEST_SEED)

    # Test multiple steps with random actions
    for i in range(500):
        # Generate valid random actions using our utility with a deterministic but different seed for each step
        iter_seed = TEST_SEED + i + 1
        random_actions = generate_valid_random_actions(environment, num_agents, seed=iter_seed)

        obs, rewards, terminated, truncated, infos = environment.step(random_actions)

        # Process episode stats if needed
        if np.any(terminated) or np.any(truncated):
            environment.process_episode_stats(infos)
            # Reset with a seed derived from the iteration to maintain determinism
            reset_seed = TEST_SEED + 1000 + i
            obs, info = environment.reset(seed=reset_seed)

    # Final verification that environment is still functioning
    assert environment._c_env is not None


def test_grid_features(environment):
    """
    Test that the grid features from the C++ environment match the expected features list.
    Ensures that the features are in the correct order and that all expected features are present.
    """
    # Expected features list based on the provided order
    expected_features = [
        "agent",
        "agent:group",
        "hp",
        "agent:frozen",
        "agent:orientation",
        "agent:color",
        "agent:inv:ore.red",
        "agent:inv:ore.blue",
        "agent:inv:ore.green",
        "agent:inv:battery",
        "agent:inv:heart",
        "agent:inv:armor",
        "agent:inv:laser",
        "agent:inv:blueprint",
        "wall",
        "swappable",
        "mine",
        "color",
        "converting",
        "inv:ore.red",
        "inv:ore.blue",
        "inv:ore.green",
        "inv:battery",
        "inv:heart",
        "inv:armor",
        "inv:laser",
        "inv:blueprint",
        "generator",
        "altar",
        "armory",
        "lasery",
        "lab",
        "factory",
        "temple",
    ]

    # Get the actual grid features from the environment
    actual_features = environment._c_env.grid_features()

    # Check that the lists have the same length
    assert len(actual_features) == len(expected_features), (
        f"Feature list length mismatch: expected {len(expected_features)}, got {len(actual_features)}"
    )

    # Check each feature individually to provide better error messages
    for i, (expected, actual) in enumerate(zip(expected_features, actual_features, strict=False)):
        assert expected == actual, f"Feature mismatch at index {i}: expected '{expected}', got '{actual}'"

    # As an additional check, verify that the lists are exactly equal
    assert actual_features == expected_features, "Feature lists don't match exactly"

    # Optionally, print the feature list for debugging (can be commented out in production)
    print("Grid features verified successfully:", actual_features)

    # Check that the number of grid features matches the number of channels in observations
    # This assumes that each feature corresponds to a channel in the observation space
    obs_shape = environment.single_observation_space.shape
    num_channels = obs_shape[2]

    # The check is relaxed to allow for the possibility that not all features are included in observations
    # or that observations might contain additional derived features
    assert 20 <= num_channels <= 50, f"Number of observation channels ({num_channels}) is outside expected range"


class TestEnvironmentFunctionality:
    """Test suite for MettaGrid environment functionality."""

    def test_timestep(self, environment):
        """Test that timestep increments correctly."""
        environment.reset()

        # Generate a valid action for all agents
        num_agents = environment.num_agents
        actions = generate_valid_random_actions(
            environment,
            num_agents,
            force_action_type=0,
            force_action_arg=0,
            seed=TEST_SEED,
        )

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

    def test_episode_stats(self, environment):
        """Test processing of episode statistics."""
        environment.reset()
        infos = {}
        environment.process_episode_stats(infos)
        # Add specific assertions if you know what should be in infos after processing

    def test_environment_properties(self, environment):
        """Test environment properties."""
        assert environment.max_steps > 0

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
        assert len(environment.action_success) == num_agents

    def test_object_type_names(self, environment):
        """Test object type names functionality."""
        assert environment.object_type_names == environment._c_env.object_type_names()
