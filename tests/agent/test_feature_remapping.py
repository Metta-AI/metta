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

        def _is_training_context(self):
            # Mock the training context detection for testing
            return getattr(self, "_mock_is_training", True)

    agent = MockAgent()

    # Test _initialize_observations with original features
    original_features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 2, "type": "scalar", "normalization": 30.0},
        "mineral": {"id": 3, "type": "scalar", "normalization": 100.0},
    }

    # Set training mode for first initialization
    agent._mock_is_training = True

    # Call _initialize_observations as a bound method
    MettaAgent._initialize_observations(agent, original_features, "cpu", agent._is_training_context())

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

    # Bind MettaAgent methods to mock agent for testing
    agent._initialize_observations = lambda features, device, is_training: MettaAgent._initialize_observations(
        agent, features, device, is_training
    )
    agent._create_feature_remapping = lambda features, is_training: MettaAgent._create_feature_remapping(
        agent, features, is_training
    )
    agent._update_normalization_factors = lambda features: MettaAgent._update_normalization_factors(agent, features)
    agent._apply_feature_remapping = lambda features, unknown_id: MettaAgent._apply_feature_remapping(
        agent, features, unknown_id
    )
    agent.get_original_feature_mapping = lambda: MettaAgent.get_original_feature_mapping(agent)

    MettaAgent._initialize_observations(agent, new_features, "cpu", True)

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

        def _is_training_context(self):
            # Mock the training context detection for testing
            return getattr(self, "_mock_is_training", True)

    agent = MockAgent()

    # First initialization with training features
    original_features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 2, "type": "scalar", "normalization": 30.0},
    }

    # Set training mode for first initialization
    agent._mock_is_training = True

    MettaAgent._initialize_observations(agent, original_features, "cpu", agent._is_training_context())

    # Create mock observation component
    class MockObsComponent:
        def update_feature_remapping(self, remap_table):
            self.remap_table = remap_table

    mock_obs = MockObsComponent()
    agent.components["_obs_"] = mock_obs

    # Bind MettaAgent methods to mock agent for testing
    agent._initialize_observations = lambda features, device, is_training: MettaAgent._initialize_observations(
        agent, features, device, is_training
    )
    agent._create_feature_remapping = lambda features, is_training: MettaAgent._create_feature_remapping(
        agent, features, is_training
    )
    agent._update_normalization_factors = lambda features: MettaAgent._update_normalization_factors(agent, features)
    agent._apply_feature_remapping = lambda features, unknown_id: MettaAgent._apply_feature_remapping(
        agent, features, unknown_id
    )
    agent.get_original_feature_mapping = lambda: MettaAgent.get_original_feature_mapping(agent)

    # Second initialization with new unknown features
    new_features_with_unknown = {
        "type_id": {"id": 5, "type": "categorical"},  # Known feature, new ID
        "hp": {"id": 7, "type": "scalar", "normalization": 35.0},  # Known feature, new ID
        "new_feature": {"id": 10, "type": "categorical"},  # Unknown feature (should be mapped to 255 in eval mode)
        "another_new": {"id": 15, "type": "scalar"},  # Unknown feature (should be mapped to 255 in eval mode)
    }

    MettaAgent._initialize_observations(agent, new_features_with_unknown, "cpu", False)  # Evaluation mode

    # Verify unknown features are mapped to 255
    assert hasattr(agent, "feature_id_remap")
    # Unknown features should be mapped to 255
    assert agent.feature_id_remap[10] == 255  # new_feature -> UNKNOWN
    assert agent.feature_id_remap[15] == 255  # another_new -> UNKNOWN
    # Known features should be remapped to their original IDs
    assert agent.feature_id_remap[5] == 0  # type_id: 5 -> 0
    assert agent.feature_id_remap[7] == 2  # hp: 7 -> 2

    # Verify the remap table in observation component
    assert mock_obs.remap_table[10] == 255  # new_feature -> UNKNOWN
    assert mock_obs.remap_table[15] == 255  # another_new -> UNKNOWN

    # Remapped features should map to their original IDs
    assert mock_obs.remap_table[5] == 0  # type_id: remapped from 5 to 0
    assert mock_obs.remap_table[7] == 2  # hp: remapped from 7 to 2

    # Feature IDs not in the current environment should map to UNKNOWN
    assert mock_obs.remap_table[0] == 255  # Original type_id ID not in current env
    assert mock_obs.remap_table[2] == 255  # Original hp ID not in current env


def test_feature_mapping_persistence_via_metadata():
    """Test that original_feature_mapping can be persisted through metadata."""

    # Create a simple mock agent to test the metadata persistence
    class MockAgent:
        def __init__(self):
            self.device = "cpu"
            self.components = {}

    agent = MockAgent()

    # Test initial feature setup
    original_features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 2, "type": "scalar", "normalization": 30.0},
        "mineral": {"id": 3, "type": "scalar", "normalization": 100.0},
    }

    # Initialize the agent
    MettaAgent._initialize_observations(agent, original_features, "cpu", True)

    # Get the original feature mapping using the new method
    agent.get_original_feature_mapping = lambda: MettaAgent.get_original_feature_mapping(agent)
    original_mapping = agent.get_original_feature_mapping()

    assert original_mapping == {"type_id": 0, "hp": 2, "mineral": 3}

    # Simulate saving to metadata and creating a new agent
    # Make a copy to avoid mutations affecting the test
    metadata = {"original_feature_mapping": original_mapping.copy()}

    # Create a new agent and restore from metadata
    new_agent = MockAgent()
    new_agent.restore_original_feature_mapping = lambda mapping: MettaAgent.restore_original_feature_mapping(
        new_agent, mapping
    )

    # Restore the mapping from metadata
    new_agent.restore_original_feature_mapping(metadata["original_feature_mapping"])

    # Verify the mapping was restored
    assert hasattr(new_agent, "original_feature_mapping")
    assert new_agent.original_feature_mapping == {"type_id": 0, "hp": 2, "mineral": 3}

    # Now test reinitialization with different feature IDs
    new_features = {
        "type_id": {"id": 5, "type": "categorical"},  # Different ID
        "hp": {"id": 7, "type": "scalar", "normalization": 30.0},  # Different ID
        "mineral": {"id": 9, "type": "scalar", "normalization": 100.0},  # Different ID
        "new_feature": {"id": 10, "type": "scalar"},  # New feature
    }

    # Create mock observation component
    class MockObsComponent:
        def update_feature_remapping(self, remap_table):
            self.remap_table = remap_table

    mock_obs = MockObsComponent()
    new_agent.components["_obs_"] = mock_obs

    # Bind MettaAgent methods to mock agent for testing
    new_agent._initialize_observations = lambda features, device, is_training: MettaAgent._initialize_observations(
        new_agent, features, device, is_training
    )
    new_agent._create_feature_remapping = lambda features, is_training: MettaAgent._create_feature_remapping(
        new_agent, features, is_training
    )
    new_agent._update_normalization_factors = lambda features: MettaAgent._update_normalization_factors(
        new_agent, features
    )
    new_agent._apply_feature_remapping = lambda features, unknown_id: MettaAgent._apply_feature_remapping(
        new_agent, features, unknown_id
    )

    # Initialize in training mode - new features should be learned
    MettaAgent._initialize_observations(new_agent, new_features, "cpu", True)

    # Verify remapping was created correctly
    assert hasattr(new_agent, "feature_id_remap")
    assert new_agent.feature_id_remap[5] == 0  # type_id: 5->0
    assert new_agent.feature_id_remap[7] == 2  # hp: 7->2
    assert new_agent.feature_id_remap[9] == 3  # mineral: 9->3

    # Verify new feature was learned (since we're in training mode)
    assert new_agent.original_feature_mapping["new_feature"] == 10

    # Test evaluation mode with unknown features
    eval_agent = MockAgent()
    eval_agent.restore_original_feature_mapping = lambda mapping: MettaAgent.restore_original_feature_mapping(
        eval_agent, mapping
    )
    # Use the original metadata, not the modified one from new_agent
    eval_agent.restore_original_feature_mapping(metadata["original_feature_mapping"])

    # Verify the restoration worked correctly
    assert eval_agent.original_feature_mapping == {"type_id": 0, "hp": 2, "mineral": 3}

    eval_agent.components["_obs_"] = MockObsComponent()
    eval_agent._create_feature_remapping = lambda features, is_training: MettaAgent._create_feature_remapping(
        eval_agent, features, is_training
    )
    eval_agent._update_normalization_factors = lambda features: MettaAgent._update_normalization_factors(
        eval_agent, features
    )
    eval_agent._apply_feature_remapping = lambda features, unknown_id: MettaAgent._apply_feature_remapping(
        eval_agent, features, unknown_id
    )

    # Initialize in eval mode - new features should map to 255
    MettaAgent._initialize_observations(eval_agent, new_features, "cpu", False)

    # In eval mode, the new feature should be mapped to UNKNOWN in the obs component
    # The feature_id_remap only contains remapped features, not all features
    obs_component = eval_agent.components["_obs_"]

    # Check if new_feature was added to feature_id_remap
    assert 10 in eval_agent.feature_id_remap, "Feature ID 10 should be in feature_id_remap"
    assert eval_agent.feature_id_remap[10] == 255, "Feature ID 10 should map to 255"

    # The remap_table is a tensor, so we need to extract the value
    assert obs_component.remap_table[10].item() == 255  # new_feature -> UNKNOWN in observation layer

    # Verify known features are remapped correctly
    assert eval_agent.feature_id_remap[5] == 0  # type_id: 5->0
    assert eval_agent.feature_id_remap[7] == 2  # hp: 7->2
    assert eval_agent.feature_id_remap[9] == 3  # mineral: 9->3
