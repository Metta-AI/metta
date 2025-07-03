import gymnasium as gym
import numpy as np
import pytest
import torch

# Import the actual class
from metta.agent.metta_agent import MettaAgent
from metta.agent.util.distribution_utils import evaluate_actions, sample_actions


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

    config_dict = {
        "clip_range": 0.1,
        "observations": {"obs_key": "grid_obs"},
        "components": {
            "_obs_": {
                "_target_": "metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper",
                "sources": None,
            },
            "obs_normalizer": {
                "_target_": "metta.agent.lib.observation_normalizer.ObservationNormalizer",
                "sources": [{"name": "_obs_"}],
            },
            "obs_flattener": {
                "_target_": "metta.agent.lib.nn_layer_library.Flatten",
                "sources": [{"name": "obs_normalizer"}],
            },
            "encoded_obs": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "obs_flattener"}],
                "nn_params": {"out_features": 64},
            },
            "_core_": {
                "_target_": "metta.agent.lib.lstm.LSTM",
                "sources": [{"name": "encoded_obs"}],
                "output_size": 64,
                "nn_params": {"num_layers": 1},
            },
            "_action_embeds_": {
                "_target_": "metta.agent.lib.action.ActionEmbedding",
                "sources": None,
                "nn_params": {"num_embeddings": 50, "embedding_dim": 8},
            },
            "actor_layer": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "_core_"}],
                "nn_params": {"out_features": 128},
            },
            "_action_": {
                "_target_": "metta.agent.lib.actor.MettaActorBig",
                "sources": [{"name": "actor_layer"}, {"name": "_action_embeds_"}],
                "bilinear_output_dim": 32,
                "mlp_hidden_dim": 128,
            },
            "critic_layer": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "_core_"}],
                "nn_params": {"out_features": 64},
                "nonlinearity": "nn.Tanh",
            },
            "_value_": {
                "_target_": "metta.agent.lib.nn_layer_library.Linear",
                "sources": [{"name": "critic_layer"}],
                "nn_params": {"out_features": 1},
                "nonlinearity": None,
            },
        },
    }

    # Create the agent with minimal config needed for the tests
    agent = MettaAgent(
        obs_space=obs_space,
        action_space=action_space,
        device="cpu",
        feature_normalizations=feature_normalizations,
        obs_width=5,
        obs_height=5,
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

        def l2_init_loss(self):
            return torch.tensor(0.0, dtype=torch.float32)

        def forward(self, x):
            return x

    # Create components for testing
    comp1 = ClippableComponent()
    comp2 = ClippableComponent()
    action_embeds = MockActionEmbeds()

    agent.components = torch.nn.ModuleDict({"_core_": comp1, "_action_": comp2, "_action_embeds_": action_embeds})

    return agent, comp1, comp2


@pytest.fixture
def metta_agent_with_actions(create_metta_agent):
    agent, _, _ = create_metta_agent
    action_names = ["action0", "action1", "action2"]
    action_max_params = [1, 2, 0]

    # Create simple test features
    features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 1, "type": "scalar", "normalization": 30.0},
        "agent:group": {"id": 2, "type": "categorical"},
    }

    # Use new interface
    agent.initialize_to_environment(features, action_names, action_max_params, "cpu")
    return agent


def test_initialize_to_environment(create_metta_agent):
    """Test the new initialize_to_environment interface."""
    agent, _, _ = create_metta_agent

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

    # Check that features were stored
    assert hasattr(agent, "active_features")
    assert agent.active_features == features

    # Check feature mappings were created
    assert hasattr(agent, "feature_id_to_name")
    assert agent.feature_id_to_name[0] == "type_id"
    assert agent.feature_id_to_name[1] == "hp"
    assert agent.feature_id_to_name[12] == "inv:ore_red"

    # Check feature normalizations
    assert agent.feature_normalizations[1] == 30.0
    assert agent.feature_normalizations[12] == 100.0

    # Check that actions were also initialized (via activate_actions)
    assert agent.action_names == action_names
    assert agent.action_max_params == action_max_params
    assert hasattr(agent, "action_index_tensor")


def test_activate_actions_backward_compatibility(create_metta_agent):
    """Test that the old activate_actions method still works for backward compatibility."""
    agent, _, _ = create_metta_agent

    action_names = ["move", "attack", "interact"]
    action_max_params = [3, 1, 2]

    # Call the old activate_actions directly
    agent.activate_actions(action_names, action_max_params, "cpu")

    # Check that actions were initialized
    assert agent.action_names == action_names
    assert agent.action_max_params == action_max_params
    assert hasattr(agent, "action_index_tensor")
    assert agent.device == "cpu"


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


def test_l2_init_loss_raises_error_for_different_shapes(create_metta_agent):
    agent, comp1, comp2 = create_metta_agent

    # Set up components to return tensors with different shapes
    comp1.l2_init_loss = lambda: torch.tensor([0.3, 0.2], dtype=torch.float32)  # tensor with shape [2]
    comp2.l2_init_loss = lambda: torch.tensor(0.5, dtype=torch.float32)  # scalar tensor

    # Verify that a RuntimeError is raised due to different tensor shapes
    with pytest.raises(RuntimeError) as excinfo:
        agent.l2_init_loss()

    # Check that the error message mentions the tensor shape mismatch
    assert "expects each tensor to be equal size" in str(excinfo.value)


def test_convert_action_to_logit_index(create_metta_agent):
    agent, _, _ = create_metta_agent

    # Setup testing environment with controlled action space
    action_names = ["action0", "action1", "action2"]
    action_max_params = [1, 2, 0]  # action0: [0,1], action1: [0,1,2], action2: [0]

    # Create simple test features
    features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 1, "type": "scalar", "normalization": 30.0},
    }

    agent.initialize_to_environment(features, action_names, action_max_params, "cpu")

    # Test single actions
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
    agent, _, _ = create_metta_agent

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
    agent, _, _ = create_metta_agent

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

    # Check that we get the original actions back
    assert torch.all(reconstructed_actions == original_actions)


def test_action_conversion_edge_cases(create_metta_agent):
    agent, _, _ = create_metta_agent

    # Setup with empty action space
    action_names = []
    action_max_params = []

    # Create simple test features
    features = {
        "type_id": {"id": 0, "type": "categorical"},
    }

    agent.initialize_to_environment(features, action_names, action_max_params, "cpu")

    # Test with empty tensor - should raise a ValueError about invalid size
    empty_actions = torch.zeros((0, 2), dtype=torch.long, device="cpu")
    with pytest.raises(
        ValueError, match=r"'flattened_action' dimension 0 \('BT'\) has invalid size 0, expected a positive value"
    ):
        agent._convert_action_to_logit_index(empty_actions)

    # Setup with single action type that has many parameters
    action_names = ["action0"]
    action_max_params = [9]  # action0: [0,1,2,3,4,5,6,7,8,9]
    agent.initialize_to_environment(features, action_names, action_max_params, "cpu")

    # Test high parameter values
    action = torch.tensor([[0, 9]], dtype=torch.long, device="cpu")  # highest valid param
    result = agent._convert_action_to_logit_index(action)
    assert result.item() == 9

    # Convert back
    logit_indices = torch.tensor([9], dtype=torch.long, device="cpu")

    result = agent._convert_logit_index_to_action(logit_indices)
    assert torch.all(result == torch.tensor([0, 9], dtype=torch.long, device="cpu"))


def test_convert_logit_index_to_action_invalid(metta_agent_with_actions):
    agent = metta_agent_with_actions
    invalid_index = agent.action_index_tensor.shape[0]
    with pytest.raises((IndexError, RuntimeError)):
        agent._convert_logit_index_to_action(torch.tensor([invalid_index], dtype=torch.long, device="cpu"))


def test_action_use(create_metta_agent):
    agent, _, _ = create_metta_agent

    # Set up action space
    action_names = ["action0", "action1", "action2"]
    action_max_params = [1, 2, 0]

    # Create simple test features
    features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 1, "type": "scalar", "normalization": 30.0},
    }

    agent.initialize_to_environment(features, action_names, action_max_params, "cpu")

    # Verify the agent correctly stored the list internally
    assert isinstance(agent.action_max_params, list)
    assert agent.action_max_params == [1, 2, 0]

    # Verify the offsets were calculated correctly
    expected_offsets = torch.tensor([0, 1, 3, 3], dtype=torch.long, device="cpu")
    assert torch.all(agent.cum_action_max_params == expected_offsets)

    # Verify action_index_tensor was created correctly
    expected_action_index = torch.tensor(
        [
            [0, 0],
            [0, 1],  # action0 with params 0, 1
            [1, 0],
            [1, 1],
            [1, 2],  # action1 with params 0, 1, 2
            [2, 0],  # action2 with param 0
        ],
        dtype=torch.long,
        device="cpu",
    )
    assert torch.all(agent.action_index_tensor == expected_action_index)

    # Test _convert_action_to_logit_index
    actions = torch.tensor(
        [
            [0, 0],  # should map to index 0
            [0, 1],  # should map to index 1
            [1, 0],  # should map to index 2
            [1, 2],  # should map to index 4
            [2, 0],  # should map to index 5
        ],
        dtype=torch.long,
        device="cpu",
    )

    expected_indices = torch.tensor([0, 1, 2, 4, 5], dtype=torch.long, device="cpu")
    action_logit_indices = agent._convert_action_to_logit_index(actions)
    assert torch.all(action_logit_indices == expected_indices)

    # Test _convert_logit_index_to_action (reverse mapping)
    reconstructed_actions = agent._convert_logit_index_to_action(expected_indices)
    assert torch.all(reconstructed_actions == actions)

    # Now let's test the distribution utils with our converted actions
    batch_size = 5
    num_total_actions = sum([param + 1 for param in action_max_params])

    # Create logits where the highest value corresponds to our test actions
    # This makes sampling deterministic for testing
    logits = torch.full((batch_size, num_total_actions), -10.0, device="cpu")

    # Make the logits corresponding to our actions very high to ensure deterministic sampling
    for i in range(batch_size):
        logits[i, expected_indices[i]] = 10.0

    # Test sample_actions (inference mode)
    sampled_indices, logprobs, entropy, log_softmax = sample_actions(logits)

    # With our strongly biased logits, sampling should return the expected indices
    assert torch.all(sampled_indices == expected_indices)

    # Verify output shapes
    assert logprobs.shape == expected_indices.shape
    assert entropy.shape == (batch_size,)
    assert log_softmax.shape == logits.shape

    # Convert sampled indices back to actions
    sampled_actions = agent._convert_logit_index_to_action(sampled_indices)
    assert torch.all(sampled_actions == actions)

    # Test evaluate_actions (training mode)
    eval_logprobs, eval_entropy, eval_log_softmax = evaluate_actions(logits, expected_indices)

    # Results should be identical to sampling since we provided the same indices
    assert torch.allclose(logprobs, eval_logprobs)
    assert torch.allclose(entropy, eval_entropy)
    assert torch.allclose(log_softmax, eval_log_softmax)

    # Test with a different batch
    batch_size2 = 3
    test_actions2 = torch.tensor(
        [
            [1, 1],  # should map to index 3
            [2, 0],  # should map to index 5
            [0, 0],  # should map to index 0
        ],
        dtype=torch.long,
        device="cpu",
    )

    expected_indices2 = torch.tensor([3, 5, 0], dtype=torch.long, device="cpu")
    batch_logit_indices = agent._convert_action_to_logit_index(test_actions2)
    assert torch.all(batch_logit_indices == expected_indices2)

    # Create logits for this batch
    logits2 = torch.full((batch_size2, num_total_actions), -10.0, device="cpu")
    for i in range(batch_size2):
        logits2[i, expected_indices2[i]] = 10.0

    # Test sampling without providing indices
    sampled_indices3, logprobs3, entropy3, log_softmax3 = sample_actions(logits2)

    # Again, with biased logits, we should get deterministic results
    assert torch.all(sampled_indices3 == expected_indices2)

    # Convert back to actions and verify
    sampled_actions2 = agent._convert_logit_index_to_action(sampled_indices3)
    assert torch.all(sampled_actions2 == test_actions2)

    # Test evaluate_actions on the same data
    eval_logprobs2, eval_entropy2, eval_log_softmax2 = evaluate_actions(logits2, expected_indices2)

    # Results should match sampling results
    assert torch.allclose(logprobs3, eval_logprobs2)
    assert torch.allclose(entropy3, eval_entropy2)
    assert torch.allclose(log_softmax3, eval_log_softmax2)

    # Finally, test the complete forward pass integration:
    # 1. Convert actions to logit indices
    # 2. Pass to sample_actions or evaluate_actions
    # 3. Convert sampled indices back to actions

    test_actions3 = torch.tensor([[0, 0], [1, 1]], dtype=torch.long, device="cpu")
    logit_indices = agent._convert_action_to_logit_index(test_actions3)

    # Create logits for deterministic sampling
    logits3 = torch.full((2, num_total_actions), -10.0, device="cpu")
    for i in range(2):
        logits3[i, logit_indices[i]] = 10.0

    # Sample with inference (like in forward when action is None)
    sampled_indices4, logprobs4, entropy4, log_softmax4 = sample_actions(logits3)

    # Verify indices match
    assert torch.all(sampled_indices4 == logit_indices)

    # Convert back to actions
    reconstructed_actions4 = agent._convert_logit_index_to_action(sampled_indices4)

    # Verify round-trip conversion
    assert torch.all(reconstructed_actions4 == test_actions3)

    # Test evaluation (like in forward when action is provided)
    eval_logprobs4, eval_entropy4, eval_log_softmax4 = evaluate_actions(logits3, logit_indices)

    # Should match sampling results
    assert torch.allclose(logprobs4, eval_logprobs4)
    assert torch.allclose(entropy4, eval_entropy4)
    assert torch.allclose(log_softmax4, eval_log_softmax4)


def test_distribution_utils_compatibility(create_metta_agent):
    """Test that sample_actions and evaluate_actions are compatible with each other."""
    agent, _, _ = create_metta_agent

    # Set up action space
    action_names = ["action0", "action1"]
    action_max_params = [2, 1]  # action0: [0,1,2], action1: [0,1]

    # Create simple test features
    features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 1, "type": "scalar", "normalization": 30.0},
    }

    agent.initialize_to_environment(features, action_names, action_max_params, "cpu")

    num_total_actions = sum([param + 1 for param in action_max_params])
    batch_size = 4

    # Create random logits
    torch.manual_seed(42)  # For reproducible tests
    logits = torch.randn(batch_size, num_total_actions)

    # Sample actions using sample_actions
    sampled_indices, sampled_logprobs, sampled_entropy, sampled_log_softmax = sample_actions(logits)

    # Evaluate the same actions using evaluate_actions
    eval_logprobs, eval_entropy, eval_log_softmax = evaluate_actions(logits, sampled_indices)

    # Results should be identical
    assert torch.allclose(sampled_logprobs, eval_logprobs, atol=1e-6)
    assert torch.allclose(sampled_entropy, eval_entropy, atol=1e-6)
    assert torch.allclose(sampled_log_softmax, eval_log_softmax, atol=1e-6)

    # Convert sampled indices to actions and back
    sampled_actions = agent._convert_logit_index_to_action(sampled_indices)
    reconstructed_indices = agent._convert_action_to_logit_index(sampled_actions)

    # Should get the same indices back
    assert torch.all(sampled_indices == reconstructed_indices)


def test_forward_training_integration(create_metta_agent):
    """Test that the forward_training method works with the new distribution utils."""
    agent, _, _ = create_metta_agent

    # Set up action space
    action_names = ["move", "use_item"]
    action_max_params = [2, 3]  # move: [0,1,2], use_item: [0,1,2,3]

    # Create simple test features
    features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 1, "type": "scalar", "normalization": 30.0},
    }

    agent.initialize_to_environment(features, action_names, action_max_params, "cpu")

    B, T = 2, 3
    num_total_actions = sum([param + 1 for param in action_max_params])  # 3 + 4 = 7

    # Create mock tensors
    value = torch.randn(B * T, 1)
    logits = torch.randn(B * T, num_total_actions)

    # Action space: move (type 0): [0,1,2], use_item (type 1): [0,1,2,3]
    action = torch.tensor(
        [
            [[0, 1], [1, 2], [0, 0]],  # batch 1
            [[1, 0], [1, 3], [0, 1]],  # batch 2
        ],
        dtype=torch.long,
        device="cpu",
    )

    # Call forward_training
    returned_action, action_log_prob, entropy, returned_value, log_probs = agent.forward_training(value, logits, action)

    # Check output shapes
    assert returned_action.shape == (B, T, 2)
    assert action_log_prob.shape == (B * T,)
    assert entropy.shape == (B * T,)
    assert returned_value.shape == (B * T, 1)
    assert log_probs.shape == (B * T, num_total_actions)

    # Check that returned action and value are the same as input
    assert torch.all(returned_action == action)
    assert torch.all(returned_value == value)

    # Additional validation: verify all actions are actually valid
    flattened_action = action.view(B * T, 2)
    for i in range(B * T):
        action_type = flattened_action[i, 0].item()
        action_param = flattened_action[i, 1].item()

        # Check action type is valid (0=move, 1=use_item)
        assert 0 <= action_type < len(action_max_params), f"Invalid action type {action_type}"

        # Check action parameter is valid for this action type
        assert isinstance(action_type, int)
        max_param = action_max_params[action_type]
        assert 0 <= action_param <= max_param, (
            f"Invalid param {action_param} for action type {action_type}, max is {max_param}"
        )
