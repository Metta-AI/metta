"""Test feature remapping functionality in MettaAgent."""

import torch
from tensordict import TensorDict

from metta.agent.lib.obs_tokenizers import ObsTokenPadStrip
from metta.agent.metta_agent import MettaAgent


# Module-level MockAgent class to avoid repetition
class MockAgent(MettaAgent):
    """Mock agent for testing feature remapping without full setup."""

    def __init__(self):
        # Skip the full MettaAgent.__init__ but set up necessary attributes
        self.device = "cpu"
        self.components = {}
        self._mock_is_training = True

    def _is_training_context(self):
        """Mock the training context detection for testing."""
        return self._mock_is_training


class MockObsComponent:
    """Mock observation component for testing remapping updates."""

    def update_feature_remapping(self, remap_table):
        self.remap_table = remap_table


def test_obs_token_pad_strip_remapping():
    """Test that ObsTokenPadStrip correctly remaps feature IDs."""
    # Create ObsTokenPadStrip with test shape
    obs_shape = (4, 3)  # 4 tokens, 3 channels
    pad_strip = ObsTokenPadStrip(obs_shape=obs_shape, name="test_pad_strip")
    pad_strip._out_tensor_shape = [0, 3]  # Set output shape

    # Create test observations [batch, tokens, 3] where 3 is [coord, feature_id, value]
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

    # Verify all remappings in a clear block
    # Note: ObsTokenPadStrip strips empty tokens (0xFF), so we only have 3 and 2 tokens respectively
    expected_feature_ids = torch.tensor(
        [
            # Batch 0 (3 non-empty tokens)
            [1, 2, 4],  # Remapped feature IDs
            # Batch 1 (2 non-empty tokens)
            [1, 2, 255],  # Last slot is padding
        ],
        dtype=torch.uint8,
    )

    # Check feature ID remapping
    assert torch.equal(remapped_obs[:, :, 1], expected_feature_ids)

    # Verify coordinates remain unchanged
    assert remapped_obs[0, 0, 0] == 0x12
    assert remapped_obs[0, 1, 0] == 0x34

    # Verify values remain unchanged
    assert remapped_obs[0, 0, 2] == 10
    assert remapped_obs[0, 1, 2] == 20


def test_feature_remapping_in_agent():
    """Test that feature remapping is correctly set up in MettaAgent."""
    agent = MockAgent()

    # Test _initialize_observations with original features
    original_features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 2, "type": "scalar", "normalization": 30.0},
        "mineral": {"id": 3, "type": "scalar", "normalization": 100.0},
    }

    # Set training mode for first initialization
    agent._mock_is_training = True
    agent._initialize_observations(original_features, "cpu", agent._is_training_context())

    # Verify original mapping stored
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

    # Add mock observation component
    mock_obs = MockObsComponent()
    agent.components["_obs_"] = mock_obs

    agent._initialize_observations(new_features, "cpu", True)

    # Verify all remappings in a clear block
    assert agent.feature_id_remap[5] == 2  # hp: 5->2
    assert agent.feature_id_remap[7] == 3  # mineral: 7->3
    # Note: type_id (0) doesn't change so it's not in the remap table

    # Verify ObsTokenPadStrip was updated with same mappings
    assert mock_obs.remap_table[5] == 2
    assert mock_obs.remap_table[7] == 3
    assert mock_obs.remap_table[0] == 0


def test_unknown_feature_handling():
    """Test that unknown features are mapped to index 255."""
    agent = MockAgent()

    # First initialization with training features
    original_features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 2, "type": "scalar", "normalization": 30.0},
    }

    # Set training mode for first initialization
    agent._mock_is_training = True
    agent._initialize_observations(original_features, "cpu", agent._is_training_context())

    # Add mock observation component
    mock_obs = MockObsComponent()
    agent.components["_obs_"] = mock_obs

    # Second initialization with one new unknown feature
    new_features_with_unknown = {
        "type_id": {"id": 5, "type": "categorical"},  # Known feature, new ID
        "hp": {"id": 7, "type": "scalar", "normalization": 35.0},  # Known feature, new ID
        "new_feature": {"id": 10, "type": "categorical"},  # Unknown feature
    }

    # Initialize in evaluation mode
    agent._initialize_observations(new_features_with_unknown, "cpu", False)

    # Verify all remappings in a clear block
    # Known features should be remapped to their original IDs
    assert agent.feature_id_remap[5] == 0  # type_id: 5 -> 0
    assert agent.feature_id_remap[7] == 2  # hp: 7 -> 2
    # Unknown feature should be mapped to 255
    assert agent.feature_id_remap[10] == 255  # new_feature -> UNKNOWN

    # Verify the remap table in observation component
    assert mock_obs.remap_table[5] == 0
    assert mock_obs.remap_table[7] == 2
    assert mock_obs.remap_table[10] == 255

    # Original feature IDs not in current env should map to UNKNOWN
    assert mock_obs.remap_table[0] == 255
    assert mock_obs.remap_table[2] == 255


def test_feature_mapping_persistence_via_metadata():
    """Test that original_feature_mapping can be persisted through metadata."""
    agent = MockAgent()

    # Test initial feature setup
    original_features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 2, "type": "scalar", "normalization": 30.0},
        "mineral": {"id": 3, "type": "scalar", "normalization": 100.0},
    }

    # Initialize the agent
    agent._initialize_observations(original_features, "cpu", True)

    # Get the original feature mapping
    original_mapping = agent.get_original_feature_mapping()
    assert original_mapping == {"type_id": 0, "hp": 2, "mineral": 3}

    # Simulate saving to metadata and creating a new agent
    metadata = {"original_feature_mapping": original_mapping.copy()}

    # Create a new agent and restore from metadata
    new_agent = MockAgent()
    new_agent.restore_original_feature_mapping(metadata["original_feature_mapping"])

    # Verify the mapping was restored
    assert new_agent.original_feature_mapping == {"type_id": 0, "hp": 2, "mineral": 3}

    # Now test reinitialization with different feature IDs
    new_features = {
        "type_id": {"id": 5, "type": "categorical"},  # Different ID
        "hp": {"id": 7, "type": "scalar", "normalization": 30.0},  # Different ID
        "mineral": {"id": 9, "type": "scalar", "normalization": 100.0},  # Different ID
        "new_feature": {"id": 10, "type": "scalar"},  # New feature
    }

    # Add mock observation component
    mock_obs = MockObsComponent()
    new_agent.components["_obs_"] = mock_obs

    # Initialize in training mode - new features should be learned
    new_agent._initialize_observations(new_features, "cpu", True)

    # Verify all remappings in a clear block
    assert new_agent.feature_id_remap[5] == 0  # type_id: 5->0
    assert new_agent.feature_id_remap[7] == 2  # hp: 7->2
    assert new_agent.feature_id_remap[9] == 3  # mineral: 9->3

    # Verify new feature was learned (since we're in training mode)
    assert new_agent.original_feature_mapping["new_feature"] == 10

    # Test evaluation mode with unknown features
    eval_agent = MockAgent()
    eval_agent.restore_original_feature_mapping(metadata["original_feature_mapping"])

    # Verify the restoration worked correctly
    assert eval_agent.original_feature_mapping == {"type_id": 0, "hp": 2, "mineral": 3}

    eval_agent.components["_obs_"] = MockObsComponent()

    # Initialize in eval mode - new features should map to 255
    eval_agent._initialize_observations(new_features, "cpu", False)

    # Check the observation component's remap table
    obs_component = eval_agent.components["_obs_"]

    # In eval mode, the new feature should be mapped to UNKNOWN
    assert eval_agent.feature_id_remap[10] == 255
    assert obs_component.remap_table[10].item() == 255  # new_feature -> UNKNOWN

    # Verify known features are remapped correctly
    assert eval_agent.feature_id_remap[5] == 0  # type_id: 5->0
    assert eval_agent.feature_id_remap[7] == 2  # hp: 7->2
    assert eval_agent.feature_id_remap[9] == 3  # mineral: 9->3
