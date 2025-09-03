import tempfile

import torch
from tensordict import TensorDict

from metta.agent.lib.obs_tokenizers import ObsTokenPadStrip
from metta.agent.mocks import MockAgent
from metta.rl.checkpoint_manager import CheckpointManager


class MockObsComponent(torch.nn.Module):
    """Mock observation component for testing remapping updates."""

    def __init__(self):
        super().__init__()
        self.remap_table = None

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
    td = TensorDict({"env_obs": observations})

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

    # Test initialize_to_environment with original features
    original_features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 2, "type": "scalar", "normalization": 30.0},
        "mineral": {"id": 3, "type": "scalar", "normalization": 100.0},
    }

    # Mock action configuration
    action_names = ["move", "attack", "gather"]
    action_max_params = [3, 2, 1]

    # Set training mode for first initialization
    agent.train()
    agent.initialize_to_environment(original_features, action_names, action_max_params, "cpu")

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

    agent.initialize_to_environment(new_features, action_names, action_max_params, "cpu")

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

    # Mock action configuration
    action_names = ["move", "attack"]
    action_max_params = [3, 2]

    # First initialization with training features
    original_features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 2, "type": "scalar", "normalization": 30.0},
    }

    # Set training mode for first initialization
    agent.train()
    agent.initialize_to_environment(original_features, action_names, action_max_params, "cpu")

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
    agent.eval()
    agent.initialize_to_environment(new_features_with_unknown, action_names, action_max_params, "cpu")

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

    # Mock action configuration
    action_names = ["move", "attack", "gather"]
    action_max_params = [3, 2, 1]

    # Test initial feature setup
    original_features = {
        "type_id": {"id": 0, "type": "categorical"},
        "hp": {"id": 2, "type": "scalar", "normalization": 30.0},
        "mineral": {"id": 3, "type": "scalar", "normalization": 100.0},
    }

    # Initialize the agent
    agent.train()
    agent.initialize_to_environment(original_features, action_names, action_max_params, "cpu")

    # Get the original feature mapping
    original_mapping = agent.get_original_feature_mapping()
    assert original_mapping == {"type_id": 0, "hp": 2, "mineral": 3}

    # Simulate saving to metadata and creating a new agent
    assert original_mapping
    metadata = {"original_feature_mapping": original_mapping.copy()}

    # Create a new agent and manually restore the mapping (simulating loaded state)
    new_agent = MockAgent()
    new_agent.original_feature_mapping = metadata["original_feature_mapping"].copy()
    # Initialize with original features
    new_agent.initialize_to_environment(original_features, action_names, action_max_params, "cpu")

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
    new_agent.train()
    new_agent.initialize_to_environment(new_features, action_names, action_max_params, "cpu")

    # Verify all remappings in a clear block
    assert new_agent.feature_id_remap[5] == 0  # type_id: 5->0
    assert new_agent.feature_id_remap[7] == 2  # hp: 7->2
    assert new_agent.feature_id_remap[9] == 3  # mineral: 9->3

    # Verify new feature was learned (since we're in training mode)
    assert new_agent.original_feature_mapping["new_feature"] == 10

    # Test evaluation mode with unknown features
    eval_agent = MockAgent()
    eval_agent.original_feature_mapping = metadata["original_feature_mapping"].copy()

    # Initialize in original features context first
    eval_agent.initialize_to_environment(original_features, action_names, action_max_params, "cpu")
    assert eval_agent.original_feature_mapping == {"type_id": 0, "hp": 2, "mineral": 3}

    eval_agent.components["_obs_"] = MockObsComponent()

    # Initialize in eval mode - new features should map to 255
    eval_agent.eval()
    eval_agent.initialize_to_environment(new_features, action_names, action_max_params, "cpu")

    # Check the observation component's remap table
    obs_component = eval_agent.components["_obs_"]

    # In eval mode, the new feature should be mapped to UNKNOWN
    assert eval_agent.feature_id_remap[10] == 255
    assert obs_component.remap_table[10].item() == 255  # new_feature -> UNKNOWN

    # Verify known features are remapped correctly
    assert eval_agent.feature_id_remap[5] == 0  # type_id: 5->0
    assert eval_agent.feature_id_remap[7] == 2  # hp: 7->2
    assert eval_agent.feature_id_remap[9] == 3  # mineral: 9->3


def test_mine_token_remapping_scenario():
    """Test the specific scenario where mines have different IDs between training runs.

    This test verifies that agents can handle when features like 'mine' have
    different token IDs between training sessions, and that new features are
    properly learned in training mode while unknown features map to 255 in eval mode.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # === PHASE 1: Initial Training with mine=1 ===
        initial_features = {
            "empty": {"id": 0, "type": "categorical"},
            "mine": {"id": 1, "type": "categorical"},  # Mine is 1
            "wall": {"id": 2, "type": "categorical"},
            "agent": {"id": 3, "type": "categorical"},
        }

        agent = MockAgent()
        agent.train()
        agent.initialize_to_environment(
            initial_features, action_names=["move", "mine"], action_max_params=[3, 0], device="cpu"
        )

        assert agent.original_feature_mapping["mine"] == 1

        # Save after initial training
        checkpoint_manager = CheckpointManager(run="mining_agent", run_dir=tmpdir)
        checkpoint_manager.save_agent(agent, epoch=1, metadata={"agent_step": 1000, "total_time": 60})

        # === PHASE 2: Continue Training with mine=2 and new 'gold' feature ===
        new_features = {
            "empty": {"id": 0, "type": "categorical"},
            "wall": {"id": 1, "type": "categorical"},  # Wall is now 1 (was 2)
            "mine": {"id": 2, "type": "categorical"},  # Mine is now 2 (was 1)
            "agent": {"id": 3, "type": "categorical"},
            "gold": {"id": 4, "type": "categorical"},  # New feature!
        }

        loaded_agent = checkpoint_manager.load_agent(epoch=1)
        mock_obs = MockObsComponent()
        loaded_agent.components["_obs_"] = mock_obs

        loaded_agent.train()  # Training mode - will learn new features
        loaded_agent.initialize_to_environment(
            new_features, action_names=["move", "mine"], action_max_params=[3, 0], device="cpu"
        )

        # Verify remapping
        assert loaded_agent.feature_id_remap[2] == 1  # mine: 2 -> 1 (remapped)
        assert loaded_agent.feature_id_remap[1] == 2  # wall: 1 -> 2 (remapped)
        assert loaded_agent.original_feature_mapping["gold"] == 4  # New feature learned

        # === PHASE 3: Evaluation with different IDs and unknown 'diamond' ===
        eval_features = {
            "empty": {"id": 10, "type": "categorical"},
            "mine": {"id": 20, "type": "categorical"},  # Mine is 20 now
            "wall": {"id": 30, "type": "categorical"},
            "agent": {"id": 40, "type": "categorical"},
            "diamond": {"id": 60, "type": "categorical"},  # Unknown feature
        }

        # Create fresh agent for eval to test unknown feature handling
        eval_agent = MockAgent()
        eval_agent.original_feature_mapping = agent.original_feature_mapping.copy()
        eval_agent.components["_obs_"] = MockObsComponent()
        eval_agent.eval()  # Eval mode

        eval_agent.initialize_to_environment(
            eval_features, action_names=["move", "mine"], action_max_params=[3, 0], device="cpu"
        )

        # Verify known features remapped correctly
        assert eval_agent.feature_id_remap[10] == 0  # empty
        assert eval_agent.feature_id_remap[20] == 1  # mine -> original ID 1
        assert eval_agent.feature_id_remap[30] == 2  # wall
        assert eval_agent.feature_id_remap[40] == 3  # agent
        assert eval_agent.feature_id_remap[60] == 255  # diamond -> UNKNOWN


def test_end_to_end_initialize_to_environment_workflow():
    """Test the full end-to-end workflow of initialize_to_environment."""
    # Create a temporary directory for saving/loading
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock environment class that provides features
        class MockMettaGridEnv:
            """Mock environment that simulates feature changes between runs."""

            def __init__(self, feature_mapping):
                self.feature_mapping = feature_mapping
                self.action_names = ["move", "turn", "interact"]
                self.max_action_args = [3, 2, 1]

            def get_observation_features(self):
                """Return features in the format expected by initialize_to_environment."""
                return {
                    name: {"id": id_val, "type": "scalar", "normalization": 10.0}
                    for name, id_val in self.feature_mapping.items()
                }

        # Step 1: Create a policy with original features
        original_env = MockMettaGridEnv(
            {
                "health": 1,
                "energy": 2,
                "gold": 3,
                "position_x": 4,
                "position_y": 5,
            }
        )

        # Create a mock policy
        policy = MockAgent()
        policy.train()  # Start in training mode

        # Initialize the policy to the original environment
        features = original_env.get_observation_features()
        policy.initialize_to_environment(features, original_env.action_names, original_env.max_action_args, "cpu")

        # Get the original feature mapping
        original_mapping = policy.get_original_feature_mapping()
        assert original_mapping == {
            "health": 1,
            "energy": 2,
            "gold": 3,
            "position_x": 4,
            "position_y": 5,
        }

        # Save using CheckpointManager - original_feature_mapping is automatically saved with the agent
        checkpoint_manager = CheckpointManager(run="test_policy", run_dir=tmpdir)

        # Create simple metadata (no need to manually save original_feature_mapping)
        metadata = {
            "agent_step": 1000,
            "epoch": 10,
            "score": 0.85,
        }

        # Save the policy - original_feature_mapping is automatically included
        checkpoint_manager.save_agent(policy, epoch=10, metadata=metadata)

        # Step 2: Load the policy in a new environment with different feature IDs
        new_env = MockMettaGridEnv(
            {
                "health": 10,  # Was 1, now 10
                "energy": 20,  # Was 2, now 20
                "gold": 30,  # Was 3, now 30
                "position_x": 40,  # Was 4, now 40
                "position_y": 50,  # Was 5, now 50
                "mana": 60,  # New feature not in original
            }
        )

        # Load the saved policy - original_feature_mapping is automatically restored
        loaded_policy = checkpoint_manager.load_agent(epoch=10)

        # Verify the original feature mapping was automatically preserved
        assert loaded_policy.get_original_feature_mapping() == original_mapping

        # Create a mock observation component to verify remapping
        mock_obs = MockObsComponent()
        loaded_policy.components["_obs_"] = mock_obs

        # Initialize to the new environment (in eval mode)
        loaded_policy.eval()  # Set to evaluation mode
        new_features = new_env.get_observation_features()
        loaded_policy.initialize_to_environment(new_features, new_env.action_names, new_env.max_action_args, "cpu")

        # Step 3: Verify the remapping was applied correctly
        # All known features should be remapped to their original IDs
        assert loaded_policy.feature_id_remap[10] == 1  # health: 10->1
        assert loaded_policy.feature_id_remap[20] == 2  # energy: 20->2
        assert loaded_policy.feature_id_remap[30] == 3  # gold: 30->3
        assert loaded_policy.feature_id_remap[40] == 4  # position_x: 40->4
        assert loaded_policy.feature_id_remap[50] == 5  # position_y: 50->5

        # Unknown feature should map to 255 in eval mode
        assert loaded_policy.feature_id_remap[60] == 255  # mana -> UNKNOWN

        # Verify the observation component received the remapping
        assert mock_obs.remap_table is not None  # Fixed: was checking tensor as boolean
        assert mock_obs.remap_table[10] == 1
        assert mock_obs.remap_table[20] == 2
        assert mock_obs.remap_table[30] == 3
        assert mock_obs.remap_table[40] == 4
        assert mock_obs.remap_table[50] == 5
        assert mock_obs.remap_table[60] == 255

        # Step 4: Test training mode with the loaded policy
        loaded_policy.train()  # Set to training mode

        # Create another environment with yet different features
        training_env = MockMettaGridEnv(
            {
                "health": 100,  # Different ID again
                "energy": 150,  # Changed from 200 to stay within 0-255
                "gold": 180,  # Changed from 300 to stay within 0-255
                "stamina": 200,  # Changed from 400 to stay within 0-255
            }
        )

        # Re-initialize in training mode
        training_features = training_env.get_observation_features()
        loaded_policy.initialize_to_environment(
            training_features, training_env.action_names, training_env.max_action_args, "cpu"
        )

        # In training mode, new features should be learned
        updated_mapping = loaded_policy.get_original_feature_mapping()
        assert "stamina" in updated_mapping
        assert updated_mapping["stamina"] == 200

        # Known features should still remap correctly
        assert loaded_policy.feature_id_remap[100] == 1  # health
        assert loaded_policy.feature_id_remap[150] == 2  # energy
        assert loaded_policy.feature_id_remap[180] == 3  # gold

        # Step 5: Verify complete workflow - train -> save -> load -> eval with remapping
        # This demonstrates the full lifecycle
        assert loaded_policy.original_feature_mapping == {
            "health": 1,
            "energy": 2,
            "gold": 3,
            "position_x": 4,
            "position_y": 5,
            "stamina": 200,  # New feature learned during training
        }
