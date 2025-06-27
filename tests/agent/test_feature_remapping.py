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
    MettaAgent._initialize_observations(agent, original_features, "cpu")

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

    # Bind _create_feature_remapping_table method to mock agent
    agent._create_feature_remapping_table = lambda features: MettaAgent._create_feature_remapping_table(agent, features)

    MettaAgent._initialize_observations(agent, new_features, "cpu")

    # Verify remapping was created
    assert hasattr(agent, "feature_id_remap")
    assert agent.feature_id_remap[5].item() == 2  # hp: 5->2
    assert agent.feature_id_remap[7].item() == 3  # mineral: 7->3

    # Verify ObsTokenPadStrip was updated
    assert hasattr(mock_obs, "remap_table")
    assert torch.equal(mock_obs.remap_table, agent.feature_id_remap)

    # Identity mappings should remain
    assert agent.feature_id_remap[0].item() == 0  # type_id unchanged
    assert agent.feature_id_remap[1].item() == 1  # unmapped feature
