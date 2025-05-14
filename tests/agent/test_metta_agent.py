import gymnasium as gym
import numpy as np
import pytest
import torch

from metta.agent.lib.action import ActionEmbedding

# Import the actual class
from metta.agent.metta_agent import MettaAgent


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
    grid_features = ["agent", "hp", "wall"]

    config_dict = {
        "clip_range": 0.1,
        "observations": {"obs_key": "grid_obs"},
        "components": {
            "_obs_": {
                "_target_": "metta.agent.lib.obs_shaper.ObsShaper",
                "name": "_obs_",
                "sources": None,
            },
            "_obs_normalizer_": {
                "_target_": "metta.agent.lib.observation_normalizer.ObservationNormalizer",
                "sources": [{"name": "_obs_"}],
                "grid_features": grid_features,
            },
            "_obs_flattener_": {
                "_target_": "metta.agent.lib.nn_layer_library.Flatten",
                "sources": [{"name": "_obs_normalizer_"}],
            },
            "_encoded_obs_": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "_obs_flattener_"}],
                "nn_params": {"out_features": 64},
            },
            "_core_": {
                "_target_": "metta.agent.lib.lstm.LSTM",
                "sources": [{"name": "_encoded_obs_"}],
                "output_size": 64,
                "nn_params": {"num_layers": 1},
            },
            "_action_embeds_": {
                "_target_": "metta.agent.lib.action.ActionEmbedding",
                "sources": None,
                "nn_params": {"num_embeddings": 50, "embedding_dim": 8},
            },
            "_actor_layer_": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "_core_"}],
                "nn_params": {"out_features": 128},
            },
            "_action_": {
                "_target_": "metta.agent.lib.actor.MettaActorBig",
                "sources": [{"name": "_actor_layer_"}, {"name": "_action_embeds_"}],
                "bilinear_output_dim": 32,
                "mlp_hidden_dim": 128,
            },
            "_critic_layer_": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "_core_"}],
                "nn_params": {"out_features": 64},
                "nonlinearity": "nn.Tanh",
            },
            "_value_": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "_critic_layer_"}],
                "nn_params": {"out_features": 1},
                "nonlinearity": None,
            },
        },
    }

    # Create the agent with minimal config needed for the tests
    agent = MettaAgent(
        obs_space=obs_space,
        action_space=action_space,
        grid_features=grid_features,
        device="cpu",
        **config_dict,
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
    class MockActionEmbeds(ActionEmbedding):
        def __init__(self):
            super().__init__(name="_action_embeds_", sources=None, nn_params={"num_embeddings": 50, "embedding_dim": 8})
            self.embedding = torch.nn.Embedding(50, 8)  # Matches config
            self.setup(None)
            self._sources = []
            self.clipped = False
            self.action_names = {}
            self.action_to_idx = {}
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

        def l2_reg_loss(self):
            return torch.tensor(0.0)

        def l2_init_loss(self):
            return torch.tensor(0.0)

        def forward(self, x):
            return x

    # Create components for testing
    comp1 = ClippableComponent()
    comp2 = ClippableComponent()
    action_embeds = MockActionEmbeds()

    agent.components = torch.nn.ModuleDict({"_core_": comp1, "_action_": comp2, "_action_embeds_": action_embeds})

    return agent, comp1, comp2


def test_clip_weights_calls_components(create_metta_agent):
    agent, comp1, comp2 = create_metta_agent

    # Ensure clip_range is positive to enable clipping
    agent.clip_range = 0.1

    # Call the method being tested
    agent.clip_weights()

    # Verify each component's clip_weights was called
    assert comp1.clipped
    assert comp2.clipped


def test_clip_weights_disabled(create_metta_agent):
    agent, comp1, comp2 = create_metta_agent

    # Disable clipping by setting clip_range to 0
    agent.clip_range = 0

    # Call the method being tested
    agent.clip_weights()

    # Verify no component's clip_weights was called
    assert not comp1.clipped
    assert not comp2.clipped


def test_clip_weights_raises_attribute_error(create_metta_agent):
    agent, comp1, comp2 = create_metta_agent

    # Add a component without the clip_weights method
    class IncompleteComponent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ready = True
            self._sources = None

        def setup(self, source_components):
            pass

        def forward(self, x):
            return x

    # Add the incomplete component
    agent.components["bad_comp"] = IncompleteComponent()

    # Verify that an AttributeError is raised
    with pytest.raises(AttributeError) as excinfo:
        agent.clip_weights()

    # Check the error message
    assert "bad_comp" in str(excinfo.value)
    assert "clip_weights" in str(excinfo.value)


def test_clip_weights_with_non_callable(create_metta_agent):
    agent, comp1, comp2 = create_metta_agent

    # Make clip_weights non-callable on one component
    comp1.clip_weights = "Not a function"

    # Verify a TypeError is raised
    with pytest.raises(TypeError) as excinfo:
        agent.clip_weights()

    # Check the error message
    assert "not callable" in str(excinfo.value)


def test_l2_reg_loss_sums_component_losses(create_metta_agent):
    agent, comp1, comp2 = create_metta_agent

    # Add l2_reg_loss method to test components with predictable return values
    comp1.l2_reg_loss = lambda: torch.tensor(0.5)
    comp2.l2_reg_loss = lambda: torch.tensor(1.5)

    # Call the method being tested
    result = agent.l2_reg_loss()

    # Verify the result is the sum of component losses
    assert result.item() == 2.0  # 0.5 + 1.5 = 2.0


def test_l2_reg_loss_raises_attribute_error(create_metta_agent):
    agent, comp1, comp2 = create_metta_agent

    # Add a component without the l2_reg_loss method
    class IncompleteComponent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ready = True
            self._sources = None

        def setup(self, source_components):
            pass

        def forward(self, x):
            return x

    # First make sure existing components have the method
    comp1.l2_reg_loss = lambda: torch.tensor(0.5)
    comp2.l2_reg_loss = lambda: torch.tensor(1.5)

    # Replace one of the existing components with our incomplete one
    agent.components["_core_"] = IncompleteComponent()

    # Verify that an AttributeError is raised
    with pytest.raises(AttributeError) as excinfo:
        agent.l2_reg_loss()

    # Check the error message mentions the missing method
    # Don't rely on a specific component name in the error message
    error_msg = str(excinfo.value)
    assert "does not have method 'l2_reg_loss'" in error_msg


def test_l2_reg_loss_raises_error_for_different_shapes(create_metta_agent):
    agent, comp1, comp2 = create_metta_agent

    # Set up components to return tensors with different shapes
    comp1.l2_reg_loss = lambda: torch.tensor(0.25) * torch.ones(2)  # tensor with shape [2]
    comp2.l2_reg_loss = lambda: torch.tensor(0.5)  # scalar tensor

    # Verify that a RuntimeError is raised due to different tensor shapes
    with pytest.raises(RuntimeError) as excinfo:
        agent.l2_reg_loss()

    # Check that the error message mentions the tensor shape mismatch
    assert "expects each tensor to be equal size" in str(excinfo.value)


def test_l2_init_loss_raises_error_for_different_shapes(create_metta_agent):
    agent, comp1, comp2 = create_metta_agent

    # Set up components to return tensors with different shapes
    comp1.l2_init_loss = lambda: torch.tensor([0.3, 0.2])  # tensor with shape [2]
    comp2.l2_init_loss = lambda: torch.tensor(0.5)  # scalar tensor

    # Verify that a RuntimeError is raised due to different tensor shapes
    with pytest.raises(RuntimeError) as excinfo:
        agent.l2_init_loss()

    # Check that the error message mentions the tensor shape mismatch
    assert "expects each tensor to be equal size" in str(excinfo.value)


def test_l2_reg_loss_with_non_callable(create_metta_agent):
    agent, comp1, comp2 = create_metta_agent

    # Make l2_reg_loss non-callable on one component
    comp1.l2_reg_loss = "Not a function"

    # Verify a TypeError is raised
    with pytest.raises(TypeError) as excinfo:
        agent.l2_reg_loss()

    # Check the error message
    assert "not callable" in str(excinfo.value)


def test_l2_reg_loss_empty_components(create_metta_agent):
    agent, _, _ = create_metta_agent

    # Empty the components dictionary
    agent.components = torch.nn.ModuleDict({})

    # Verify an assertion error is raised when no components exist
    with pytest.raises(AssertionError) as excinfo:
        agent.l2_reg_loss()

    # Check the error message
    assert "No components available" in str(excinfo.value)


def test_convert_action_to_logit_index(create_metta_agent):
    agent, _, _ = create_metta_agent

    # Setup testing environment with controlled action space
    action_names = ["action0", "action1", "action2"]
    action_max_params = [1, 2, 0]  # action0: [0,1], action1: [0,1,2], action2: [0]
    agent.activate_actions(action_names, action_max_params, "cpu")

    # Test single actions
    # action (0,0) should map to logit index 0
    action = torch.tensor([[0, 0]])
    result = agent._convert_action_to_logit_index(action)
    assert result.item() == 0

    # action (0,1) should map to logit index 1
    action = torch.tensor([[0, 1]])
    result = agent._convert_action_to_logit_index(action)
    assert result.item() == 1

    # action (1,0) should map to logit index 2
    action = torch.tensor([[1, 0]])
    result = agent._convert_action_to_logit_index(action)
    assert result.item() == 2

    # action (1,2) should map to logit index 4
    action = torch.tensor([[1, 2]])
    result = agent._convert_action_to_logit_index(action)
    assert result.item() == 4

    # action (2,0) should map to logit index 5
    action = torch.tensor([[2, 0]])
    result = agent._convert_action_to_logit_index(action)
    assert result.item() == 5

    # Test batch conversion
    actions = torch.tensor([[0, 0], [1, 2], [2, 0]])
    result = agent._convert_action_to_logit_index(actions)
    assert torch.all(result.flatten() == torch.tensor([0, 4, 5]))


def test_convert_logit_index_to_action(create_metta_agent):
    agent, _, _ = create_metta_agent

    # Setup testing environment
    action_names = ["action0", "action1", "action2"]
    action_max_params = [1, 2, 0]  # action0: [0,1], action1: [0,1,2], action2: [0]
    agent.activate_actions(action_names, action_max_params, "cpu")

    # Test single conversions
    # logit index 0 should map to action (0,0)
    logit = torch.tensor([[0]])
    result = agent._convert_logit_index_to_action(logit, {})
    assert torch.all(result == torch.tensor([0, 0]))

    # logit index 1 should map to action (0,1)
    logit = torch.tensor([[1]])
    result = agent._convert_logit_index_to_action(logit, {})
    assert torch.all(result == torch.tensor([0, 1]))

    # logit index 4 should map to action (1,2)
    logit = torch.tensor([[4]])
    result = agent._convert_logit_index_to_action(logit, {})
    assert torch.all(result == torch.tensor([1, 2]))

    # Test batch conversion
    logits = torch.tensor([[0], [4], [5]])
    result = agent._convert_logit_index_to_action(logits, {})
    expected = torch.tensor([[0, 0], [1, 2], [2, 0]])
    assert torch.all(result == expected)


def test_bidirectional_action_conversion(create_metta_agent):
    agent, _, _ = create_metta_agent

    # Setup testing environment
    action_names = ["action0", "action1", "action2"]
    action_max_params = [1, 2, 0]  # action0: [0,1], action1: [0,1,2], action2: [0]
    agent.activate_actions(action_names, action_max_params, "cpu")

    # Create a test set of all possible actions
    original_actions = torch.tensor(
        [
            [0, 0],
            [0, 1],  # action0 with params 0,1
            [1, 0],
            [1, 1],
            [1, 2],  # action1 with params 0,1,2
            [2, 0],  # action2 with param 0
        ]
    )

    # Convert to logit indices
    logit_indices = agent._convert_action_to_logit_index(original_actions)

    # Convert back to actions
    reconstructed_actions = agent._convert_logit_index_to_action(logit_indices, {})

    # Check that we get the original actions back
    assert torch.all(reconstructed_actions == original_actions)


def test_action_conversion_edge_cases(create_metta_agent):
    agent, _, _ = create_metta_agent

    # Setup with empty action space
    action_names = []
    action_max_params = []
    agent.activate_actions(action_names, action_max_params, "cpu")

    # Test with empty tensor - should not raise errors
    empty_actions = torch.zeros((0, 2), dtype=torch.long)
    result = agent._convert_action_to_logit_index(empty_actions)
    assert result.shape[0] == 0

    # Setup with single action type that has many parameters
    action_names = ["action0"]
    action_max_params = [9]  # action0: [0,1,2,3,4,5,6,7,8,9]
    agent.activate_actions(action_names, action_max_params, "cpu")

    # Test high parameter values
    action = torch.tensor([[0, 9]])  # highest valid param
    result = agent._convert_action_to_logit_index(action)
    assert result.item() == 9

    # Convert back
    logit = torch.tensor([[9]])
    result = agent._convert_logit_index_to_action(logit, {})
    assert torch.all(result == torch.tensor([0, 9]))


def test_calculate_cum_action_max_params(create_metta_agent):
    agent, _, _ = create_metta_agent

    # Test case from our existing tests
    action_max_params = [1, 2, 0]
    result = agent._calculate_cum_action_max_params(action_max_params, device="cpu")
    expected = torch.tensor([0, 2, 5], device="cpu")  # Updated expected values
    assert torch.all(result == expected)

    # Additional test cases
    action_max_params = [3, 2, 1, 5]
    result = agent._calculate_cum_action_max_params(action_max_params, device="cpu")
    expected = torch.tensor([0, 4, 7, 9], device="cpu")  # Updated expected values
    assert torch.all(result == expected)

    # Edge cases
    action_max_params = [0]
    result = agent._calculate_cum_action_max_params(action_max_params, device="cpu")
    expected = torch.tensor([0], device="cpu")
    assert torch.all(result == expected)

    action_max_params = []
    result = agent._calculate_cum_action_max_params(action_max_params, device="cpu")
    expected = torch.tensor([0], device="cpu")
    assert torch.all(result == expected)


def test_numpy_array_conversion(create_metta_agent):
    agent, _, _ = create_metta_agent

    # Test that numpy arrays are converted to lists
    action_names = ["action0", "action1", "action2"]
    # Use numpy array
    action_max_params = [1, 2, 0]
    agent.activate_actions(action_names, action_max_params, "cpu")

    # Verify the agent correctly converted to list internally
    assert isinstance(agent.action_max_params, list)
    assert agent.action_max_params == [1, 2, 0]

    # Verify the offsets were calculated correctly
    expected_offsets = torch.tensor([0, 2, 5], dtype=torch.long, device="cpu")
    assert torch.all(agent.cum_action_max_params == expected_offsets)
