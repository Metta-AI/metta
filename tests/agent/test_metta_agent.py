import gymnasium as gym
import numpy as np
import pytest
import torch

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
                "sources": None,  # Start with None, not circular reference
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
        obs_space=obs_space, action_space=action_space, grid_features=grid_features, device="cpu", **config_dict
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

    # Replace the agent's components with our test components
    comp1 = ClippableComponent()
    comp2 = ClippableComponent()
    agent.components = torch.nn.ModuleDict({"_core_": comp1, "_action_": comp2})

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
