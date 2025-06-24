import numpy as np
import pytest
from gymnasium.spaces import Box, MultiDiscrete
from omegaconf import OmegaConf

from mettagrid.curriculum import SingleTaskCurriculum
from mettagrid.mettagrid_env import MettaGridEnv, dtype_actions
from mettagrid.util.hydra import get_cfg


@pytest.fixture
def environment():
    """Create and initialize the environment."""

    cfg = get_cfg("benchmark")
    print(OmegaConf.to_yaml(cfg))

    curriculum = SingleTaskCurriculum("benchmark", cfg)
    env = MettaGridEnv(curriculum, render_mode="human")
    env.reset()
    yield env
    # Cleanup after test
    del env


@pytest.mark.parametrize(
    "dependency",
    [
        "hydra",
        "matplotlib",
        "pettingzoo",
        "pytest",
        "yaml",
        "rich",
        "scipy",
        "tabulate",
        "termcolor",
        "wandb",
        "pandas",
        "tqdm",
    ],
)
def test_dependency_import(dependency):
    """Test that individual dependencies can be imported."""
    try:
        __import__(dependency)
    except ImportError as e:
        pytest.fail(f"Failed to import {dependency}: {str(e)}")


class TestEnvironmentFunctionality:
    """Test suite for MettaGrid environment functionality."""

    def test_env_initialization(self, environment):
        """Test environment initialization."""
        assert environment._c_env is not None
        assert environment._grid_env is not None
        assert environment._c_env == environment._grid_env
        assert environment.done is False

    def test_env_reset(self, environment):
        """Test environment reset functionality."""
        # Reset should return observation and info
        obs, info = environment.reset()

        # Check observation structure
        [num_agents, grid_width, grid_height, num_channels] = obs.shape
        num_expected_agents = environment._c_env.num_agents
        assert num_agents == num_expected_agents
        assert grid_width > 0
        assert grid_height > 0
        assert 20 <= num_channels <= 50

    def test_env_step(self, environment):
        """Test environment step functionality."""
        environment.reset()

        # Check initial timestep
        assert environment._c_env.current_step == 0

        num_agents = environment._c_env.num_agents
        # Take a step with NoOp actions for all agents
        actions = np.array([[0, 0]] * num_agents, dtype=dtype_actions)
        (obs, rewards, terminated, truncated, infos) = environment.step(actions)

        # Check timestep increased
        assert environment._c_env.current_step == 1

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

        # Check observation space (Box)
        observation_space = environment.single_observation_space

        assert isinstance(observation_space, Box), f"Expected Box observation space, got {type(observation_space)}"
        observation_space_shape = observation_space.shape
        assert len(observation_space_shape) == 3, (
            f"Expected 3D observation space, got {len(observation_space_shape)}D: {observation_space_shape}"
        )
        assert observation_space_shape[0] > 0, f"grid width: {observation_space_shape[0]}"
        assert observation_space_shape[1] > 0, f"grid height: {observation_space_shape[1]}"
        assert observation_space_shape[2] > 0, f"channels: {observation_space_shape[2]}"

        # Check action space (MultiDiscrete)
        action_space = environment.single_action_space
        assert isinstance(action_space, MultiDiscrete), f"Expected MultiDiscrete action space, got {type(action_space)}"
        action_space_shape = action_space.shape
        assert len(action_space_shape) == 1, (
            f"Expected 1D action space, got {len(action_space_shape)}D: {action_space_shape}"
        )
        assert action_space_shape[0] > 0, f"number of discrete actions: {action_space_shape[0]}"

        # Check env properties
        assert environment._c_env.map_width > 0
        assert environment._c_env.map_height > 0
        num_agents = environment._c_env.num_agents
        assert num_agents > 0
        assert len(environment.action_success) == num_agents

    def test_object_type_names(self, environment):
        """Test object type names functionality."""
        assert environment.object_type_names == environment._c_env.object_type_names()
