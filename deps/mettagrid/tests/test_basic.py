import pytest
from omegaconf import OmegaConf

from mettagrid.config.utils import get_cfg
from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.resolvers import register_resolvers


@pytest.fixture
def environment():
    """Create and initialize the environment."""

    register_resolvers()

    cfg = get_cfg("benchmark")
    print(OmegaConf.to_yaml(cfg))

    env = MettaGridEnv(cfg, render_mode="human", _recursive_=False)
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
        "raylib",
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


@pytest.mark.parametrize(
    "module_name",
    [
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
    ],
)
def test_mettagrid_module_import(module_name):
    """Test that individual mettagrid modules can be imported."""
    try:
        __import__(module_name)
    except ImportError as e:
        pytest.fail(f"Failed to import {module_name}: {str(e)}")


class TestEnvironmentFunctionality:
    """Test suite for MettaGrid environment functionality."""

    def test_env_initialization(self, environment):
        """Test environment initialization."""
        assert environment._renderer is None
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
        num_expected_agents = environment._c_env.num_agents()
        assert num_agents == num_expected_agents
        assert grid_width > 0
        assert grid_height > 0
        assert 20 <= num_channels <= 50

    def test_env_step(self, environment):
        """Test environment step functionality."""
        environment.reset()

        # Check initial timestep
        assert environment._c_env.current_timestep() == 0

        num_agents = environment._c_env.num_agents()
        # Take a step with NoOp actions for all agents
        (obs, rewards, terminated, truncated, infos) = environment.step([[0, 0]] * num_agents)

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

    def test_object_type_names(self, environment):
        """Test object type names functionality."""
        assert environment.object_type_names() == environment._c_env.object_type_names()
