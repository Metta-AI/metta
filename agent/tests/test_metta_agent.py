import gymnasium as gym
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite

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
    # Use the current interface but with the agent the old tests expected
    agent_cfg = AgentConfig(name="fast")

    # Create the agent with the CURRENT signature
    agent = MettaAgent(
        env=MinimalEnv(),
        system_cfg=system_cfg,
        policy_architecture_cfg=agent_cfg,
        policy=None,  # Will create ComponentPolicy internally
    )

    # Create test components that have clip_weights method for testing
    class ClippableComponent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.ready = True
            self._sources = None
            self.clipped = False

        def setup(self, source_components):
            pass

        def clip_weights(self):
            # This is a mock implementation for testing
            self.clipped = True
            return True

        def forward(self, x):
            return x

    # Create a mock ActionEmbedding component that has the activate_actions method
    class MockActionEmbeds(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(50, 8)  # Matches config
            self.ready = True
            self._sources = None
            self.clipped = False
            self.action_names = None
            self.device = None

        def setup(self, source_components):
            pass

        def clip_weights(self):
            self.clipped = True
            return True

        def activate_actions(self, action_names, device):
            self.action_names = action_names
            self.device = device
            # Create a simple mapping that will let us test action conversions
            self.action_to_idx = {name: i for i, name in enumerate(action_names)}

        def initialize_to_environment(self, action_names, device):
            # Simple implementation that just calls activate_actions
            self.activate_actions(action_names, device)

        def l2_init_loss(self):
            return torch.tensor(0.0, dtype=torch.float32)

        def forward(self, x):
            return x

    # Create components for testing
    comp1 = ClippableComponent()
    comp2 = ClippableComponent()
    action_embeds = MockActionEmbeds()

    # Set components on the policy, not the agent
    if hasattr(agent.policy, "components"):
        agent.policy.components = torch.nn.ModuleDict(
            {"_core_": comp1, "_action_": comp2, "_action_embeds_": action_embeds}
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
    assert isinstance(spec, Composite)

    # Check it has expected keys (updated for simplified architecture)
    assert "env_obs" in spec
    assert "dones" in spec


def test_clip_weights(create_metta_agent):
    """Test that clip_weights can be called without error."""
    agent = create_metta_agent

    # This should not raise an error
    agent.clip_weights()

    # Check the method exists and is callable
    assert callable(getattr(agent, "clip_weights", None))


def test_l2_init_loss(create_metta_agent):
    """Test that l2_init_loss returns a tensor."""
    agent = create_metta_agent

    loss = agent.l2_init_loss()

    # Check it returns a tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dtype == torch.float32


def test_convert_action_to_logit_index(create_metta_agent):
    """Test the critical action conversion functionality from old tests."""
    agent = create_metta_agent

    # Setup testing environment with controlled action space
    action_names = ["action0", "action1", "action2"]
    action_max_params = [1, 2, 0]  # action0: [0,1], action1: [0,1,2], action2: [0]

    # Create simple test features
    features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 1, "type": "scalar", "normalization": 30.0},
    }

    agent.initialize_to_environment(features, action_names, action_max_params, "cpu")

    # Test single actions (from old comprehensive tests)
    # action (0,0) should map to logit index 0
    action = torch.tensor([[0, 0]], dtype=torch.long, device="cpu")
    result = agent._convert_action_to_logit_index(action)
    assert result.item() == 0

    # action (0,1) should map to logit index 1
    action = torch.tensor([[0, 1]], dtype=torch.long, device="cpu")
    result = agent._convert_action_to_logit_index(action)
    assert result.item() == 1

    # action (1,0) should map to logit index 2
    action = torch.tensor([[1, 0]], dtype=torch.long, device="cpu")
    result = agent._convert_action_to_logit_index(action)
    assert result.item() == 2

    # action (1,2) should map to logit index 4
    action = torch.tensor([[1, 2]], dtype=torch.long, device="cpu")
    result = agent._convert_action_to_logit_index(action)
    assert result.item() == 4

    # action (2,0) should map to logit index 5
    action = torch.tensor([[2, 0]], dtype=torch.long, device="cpu")
    result = agent._convert_action_to_logit_index(action)
    assert result.item() == 5

    # Test batch conversion
    actions = torch.tensor([[0, 0], [1, 2], [2, 0]], dtype=torch.long, device="cpu")
    result = agent._convert_action_to_logit_index(actions)
    assert torch.all(result.flatten() == torch.tensor([0, 4, 5], dtype=torch.long, device="cpu"))


def test_convert_logit_index_to_action(create_metta_agent):
    """Test the reverse action conversion functionality."""
    agent = create_metta_agent

    # Setup testing environment
    action_names = ["action0", "action1", "action2"]
    action_max_params = [1, 2, 0]  # action0: [0,1], action1: [0,1,2], action2: [0]

    # Create simple test features
    features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 1, "type": "scalar", "normalization": 30.0},
    }

    agent.initialize_to_environment(features, action_names, action_max_params, "cpu")

    # Test single conversions
    # logit index 0 should map to action (0,0)
    logit_indices = torch.tensor([0], dtype=torch.long, device="cpu")
    result = agent._convert_logit_index_to_action(logit_indices)
    assert torch.all(result == torch.tensor([0, 0], dtype=torch.long, device="cpu"))

    # logit index 1 should map to action (0,1)
    logit_indices = torch.tensor([1], dtype=torch.long, device="cpu")
    result = agent._convert_logit_index_to_action(logit_indices)
    assert torch.all(result == torch.tensor([0, 1], dtype=torch.long, device="cpu"))

    # logit index 4 should map to action (1,2)
    logit_indices = torch.tensor([4], dtype=torch.long, device="cpu")
    result = agent._convert_logit_index_to_action(logit_indices)
    assert torch.all(result == torch.tensor([1, 2], dtype=torch.long, device="cpu"))

    # Test batch conversion
    logit_indices = torch.tensor([0, 4, 5], dtype=torch.long, device="cpu")
    result = agent._convert_logit_index_to_action(logit_indices)
    expected = torch.tensor([[0, 0], [1, 2], [2, 0]], dtype=torch.long, device="cpu")
    assert torch.all(result == expected)


def test_bidirectional_action_conversion(create_metta_agent):
    """Test that action conversion is bidirectional (critical for training)."""
    agent = create_metta_agent

    # Setup testing environment
    action_names = ["action0", "action1", "action2"]
    action_max_params = [1, 2, 0]  # action0: [0,1], action1: [0,1,2], action2: [0]

    # Create simple test features
    features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 1, "type": "scalar", "normalization": 30.0},
    }

    agent.initialize_to_environment(features, action_names, action_max_params, "cpu")

    # Create a test set of all possible actions
    original_actions = torch.tensor(
        [
            [0, 0],
            [0, 1],  # action0 with params 0,1
            [1, 0],
            [1, 1],
            [1, 2],  # action1 with params 0,1,2
            [2, 0],  # action2 with param 0
        ],
        dtype=torch.long,
        device="cpu",
    )

    # Convert to logit indices
    logit_indices = agent._convert_action_to_logit_index(original_actions)

    # Convert back to actions
    reconstructed_actions = agent._convert_logit_index_to_action(logit_indices)

    # Check that we get the original actions back (critical!)
    assert torch.all(reconstructed_actions == original_actions)
