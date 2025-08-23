import gymnasium as gym
import numpy as np
import pytest
import torch
from tensordict import TensorDict

# Import the actual class
from metta.agent.agent_config import AgentConfig
from metta.agent.metta_agent import MettaAgent
from metta.rl.system_config import SystemConfig


@pytest.fixture
def create_metta_agent():
    # Create minimal observation and action spaces for testing
    obs_space = gym.spaces.Dict(
        {
            "grid_obs": gym.spaces.Box(
                low=0,
                high=1,
                shape=(3, 5, 5, 3),  # (batch, width, height, features)
                dtype=np.float32,
            ),
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )

    action_space = gym.spaces.MultiDiscrete([3, 2])
    feature_normalizations = {0: 1.0, 1: 30.0, 2: 10.0}

    # Create a minimal environment mock
    class MinimalEnv:
        def __init__(self):
            self.single_observation_space = obs_space["grid_obs"]
            self.obs_width = 5
            self.obs_height = 5
            self.single_action_space = action_space
            self.feature_normalizations = feature_normalizations

    # Create system config
    system_cfg = SystemConfig(device="cpu")
    # Create a proper AgentConfig object
    agent_cfg = AgentConfig(name="fast")

    # Create the agent
    agent = MettaAgent(
        env=MinimalEnv(),
        system_cfg=system_cfg,
        policy_architecture_cfg=agent_cfg,
        policy=None,  # Will create ComponentPolicy internally
    )

    return agent


def test_basic_agent_creation(create_metta_agent):
    """Test that we can create a basic agent."""
    agent = create_metta_agent

    # Check basic attributes
    assert agent.device == "cpu"
    assert agent.obs_width == 5
    assert agent.obs_height == 5
    assert agent.policy is not None
    assert agent.cfg.name == "fast"


@pytest.mark.skip(reason="Forward pass requires full environment setup, not suitable for unit testing")
def test_forward_pass_with_tensordict(create_metta_agent):
    """Test the forward pass through the agent with TensorDict.

    This test is skipped because the ComponentPolicy architecture requires
    properly sized observations that match the network's architectural assumptions.
    Full forward pass testing should be done with integration tests using real environments.
    """
    pass


def test_initialize_to_environment(create_metta_agent):
    """Test the initialize_to_environment interface."""
    agent = create_metta_agent

    # Create test features dictionary
    features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 1, "type": "scalar", "normalization": 30.0},
        "agent:group": {"id": 2, "type": "categorical"},
        "inv:ore_red": {"id": 12, "type": "scalar", "normalization": 100.0},
    }

    action_names = ["move", "attack", "interact"]
    action_max_params = [3, 1, 2]

    # Call initialize_to_environment
    agent.initialize_to_environment(features, action_names, action_max_params, "cpu")

    # Check feature mappings were created
    assert hasattr(agent, "feature_id_to_name")
    assert agent.feature_id_to_name[0] == "type_id"
    assert agent.feature_id_to_name[1] == "hp"
    assert agent.feature_id_to_name[12] == "inv:ore_red"

    # Check feature normalizations
    assert agent.feature_normalizations[1] == 30.0
    assert agent.feature_normalizations[12] == 100.0

    # Check that actions were also initialized
    assert agent.action_names == action_names
    assert agent.action_max_params == action_max_params
    assert hasattr(agent, "action_index_tensor")


def test_activate_actions_via_initialize(create_metta_agent):
    """Test that actions are properly initialized through initialize_to_environment."""
    agent = create_metta_agent

    # Create minimal features for initialization
    features = {"type_id": {"id": 0, "type": "categorical"}}
    action_names = ["move", "attack", "interact"]
    action_max_params = [3, 1, 2]

    # Initialize the environment (which sets up both features and actions)
    agent.initialize_to_environment(features, action_names, action_max_params, "cpu")

    # Check that actions were initialized
    assert agent.action_names == action_names
    assert agent.action_max_params == action_max_params
    assert hasattr(agent, "action_index_tensor")
    assert agent.device == "cpu"


def test_policy_none_error():
    """Test that error is raised when policy is None and forward is called."""
    # Create agent with no policy
    system_cfg = SystemConfig(device="cpu")
    agent_cfg = AgentConfig(name="fast")

    class MinimalEnv:
        def __init__(self):
            self.single_observation_space = gym.spaces.Box(low=0, high=1, shape=(5, 5, 3), dtype=np.float32)
            self.obs_width = 5
            self.obs_height = 5
            self.single_action_space = gym.spaces.MultiDiscrete([3, 2])
            self.feature_normalizations = {0: 1.0}

    # Create agent but force policy to None
    agent = MettaAgent(env=MinimalEnv(), system_cfg=system_cfg, policy_architecture_cfg=agent_cfg, policy=None)
    agent.policy = None  # Force it to None for testing

    # Try forward pass - should raise error
    obs = TensorDict(
        {"grid_obs": torch.randn(1, 5, 5, 3), "global_vars": torch.zeros(1, 0, dtype=torch.int32)}, batch_size=1
    )

    with pytest.raises(RuntimeError, match="No policy set"):
        agent(obs)


def test_device_handling(create_metta_agent):
    """Test that device is properly set on agent and policy."""
    agent = create_metta_agent

    # Check device is set
    assert agent.device == "cpu"

    # Check policy device if policy supports it
    if hasattr(agent.policy, "device"):
        assert agent.policy.device == "cpu"


def test_agent_experience_spec(create_metta_agent):
    """Test getting the agent experience specification."""
    agent = create_metta_agent

    # Get spec
    spec = agent.get_agent_experience_spec()

    # Check it's a Composite spec
    from torchrl.data import Composite

    assert isinstance(spec, Composite)
