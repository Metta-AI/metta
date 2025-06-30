"""Test feature remapping functionality in MettaAgent."""

import torch
from tensordict import TensorDict

from metta.agent.lib.obs_tokenizers import ObsTokenPadStrip
from metta.agent.metta_agent import MettaAgent


def test_obs_token_pad_strip_remapping():
    """Test that ObsTokenPadStrip correctly remaps feature IDs."""
    # Create ObsTokenPadStrip with test shape
    obs_shape = (4, 3)  # 4 tokens, 3 channels
    pad_strip = ObsTokenPadStrip(obs_shape=obs_shape, name="test_pad_strip")
    pad_strip._out_tensor_shape = [0, 3]  # Set output shape

    # Create test observations [batch, tokens, 3] where 3 is [coord, feature_id, value]
    batch_size = 2
    observations = torch.tensor(
        [
            # Batch 0
            [
                [0x12, 3, 10],  # Feature ID 3
                [0x34, 5, 20],  # Feature ID 5
                [0x56, 7, 30],  # Feature ID 7
                [0xFF, 0xFF, 0xFF],  # Empty token
            ],
            # Batch 1
            [
                [0x78, 3, 40],  # Feature ID 3
                [0x9A, 5, 50],  # Feature ID 5
                [0xFF, 0xFF, 0xFF],  # Empty token
                [0xFF, 0xFF, 0xFF],  # Empty token
            ],
        ],
        dtype=torch.uint8,
    )

    # Create remapping: 3->1, 5->2, 7->4
    remap_table = torch.arange(256, dtype=torch.uint8)
    remap_table[3] = 1
    remap_table[5] = 2
    remap_table[7] = 4
    pad_strip.update_feature_remapping(remap_table)

    # Create input tensor dict
    td = TensorDict({"x": observations})

    # Process through pad strip
    output_td = pad_strip._forward(td)
    remapped_obs = output_td["test_pad_strip"]

    # Verify remapping
    assert remapped_obs[0, 0, 1] == 1  # 3->1
    assert remapped_obs[0, 1, 1] == 2  # 5->2
    assert remapped_obs[0, 2, 1] == 4  # 7->4
    assert remapped_obs[1, 0, 1] == 1  # 3->1
    assert remapped_obs[1, 1, 1] == 2  # 5->2

    # Verify other fields remain unchanged
    assert remapped_obs[0, 0, 0] == 0x12  # Coords unchanged
    assert remapped_obs[0, 1, 0] == 0x34
    assert remapped_obs[0, 0, 2] == 10  # Values unchanged
    assert remapped_obs[0, 1, 2] == 20


def test_feature_remapping_in_agent():
    """Test that feature remapping is correctly set up in MettaAgent."""

    # Create a simple mock agent to test feature remapping without full setup
    class MockAgent:
        def __init__(self):
            self.device = "cpu"
            self.components = {}

    agent = MockAgent()

    # Test _initialize_observations with original features
    original_features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 2, "type": "scalar", "normalization": 30.0},
        "mineral": {"id": 3, "type": "scalar", "normalization": 100.0},
    }

    # Call _initialize_observations as a bound method
    MettaAgent._initialize_observations(agent, original_features, "cpu", is_training=True)

    # Verify original mapping stored
    assert hasattr(agent, "original_feature_mapping")
    assert agent.original_feature_mapping == {
        "type_id": 0,
        "hp": 2,
        "mineral": 3,
    }

    # Test reinitialization with different feature IDs
    new_features = {
        "type_id": {"id": 0, "type": "categorical"},  # Same
        "hp": {"id": 5, "type": "scalar", "normalization": 30.0},  # Was 2, now 5
        "mineral": {"id": 7, "type": "scalar", "normalization": 100.0},  # Was 3, now 7
    }

    # Create a mock _obs_ component
    class MockObsComponent:
        def update_feature_remapping(self, remap_table):
            self.remap_table = remap_table

    mock_obs = MockObsComponent()
    agent.components["_obs_"] = mock_obs

    # Bind required methods to mock agent
    agent._create_feature_remapping = lambda features: MettaAgent._create_feature_remapping(agent, features)
    agent._update_normalization_factors = lambda features: MettaAgent._update_normalization_factors(agent, features)
    agent.is_training = True  # Set the training flag

    MettaAgent._initialize_observations(agent, new_features, "cpu", is_training=True)

    # Verify remapping was created
    assert hasattr(agent, "feature_id_remap")
    assert agent.feature_id_remap[5] == 2  # hp: 5->2
    assert agent.feature_id_remap[7] == 3  # mineral: 7->3

    # Verify ObsTokenPadStrip was updated
    assert hasattr(mock_obs, "remap_table")
    assert mock_obs.remap_table[5] == 2  # hp: 5->2
    assert mock_obs.remap_table[7] == 3  # mineral: 7->3
    assert mock_obs.remap_table[0] == 0  # type_id unchanged


def test_unknown_feature_handling():
    """Test that unknown features are mapped to index 255."""

    # Create a simple mock agent
    class MockAgent:
        def __init__(self):
            self.device = "cpu"
            self.components = {}

    agent = MockAgent()

    # First initialization with training features
    original_features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 2, "type": "scalar", "normalization": 30.0},
    }

    MettaAgent._initialize_observations(agent, original_features, "cpu", is_training=True)

    # Create mock observation component
    class MockObsComponent:
        def update_feature_remapping(self, remap_table):
            self.remap_table = remap_table

    mock_obs = MockObsComponent()
    agent.components["_obs_"] = mock_obs

    # Bind required methods to mock agent
    agent._create_feature_remapping = lambda features: MettaAgent._create_feature_remapping(agent, features)
    agent._update_normalization_factors = lambda features: MettaAgent._update_normalization_factors(agent, features)
    agent.is_training = False  # Set to evaluation mode to test unknown feature handling

    # Second initialization with new unknown features
    new_features = {
        "type_id": {"id": 0, "type": "categorical"},  # Known
        "hp": {"id": 2, "type": "scalar", "normalization": 30.0},  # Known
        "new_feature": {"id": 10, "type": "scalar"},  # Unknown!
        "another_new": {"id": 15, "type": "categorical"},  # Unknown!
    }

    MettaAgent._initialize_observations(agent, new_features, "cpu", is_training=False)

    # Verify unknown features are mapped to 255
    assert hasattr(agent, "feature_id_remap")
    assert agent.feature_id_remap[10] == 255  # new_feature -> UNKNOWN
    assert agent.feature_id_remap[15] == 255  # another_new -> UNKNOWN

    # Verify the remap table in observation component
    assert mock_obs.remap_table[10] == 255
    assert mock_obs.remap_table[15] == 255

    # Known features should map to themselves
    assert mock_obs.remap_table[0] == 0
    assert mock_obs.remap_table[2] == 2
