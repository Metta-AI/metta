#!/usr/bin/env python3
"""Test that PolicyStore can save/load policies"""

import os
import tempfile

import torch
import torch.nn as nn
from omegaconf import OmegaConf


# Create a minimal policy class that mimics MettaAgent structure
class MinimalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.components = nn.ModuleDict(
            {
                "_core_": nn.LSTM(10, 10),
                "_value_": nn.Linear(10, 1),
                "_action_": nn.Linear(10, 5),
            }
        )

    def forward(self, x):
        return self.fc(x)


def test_policy_save_load_without_pydantic():
    """Test that we can save and load a policy without pydantic errors"""

    # Import PolicyStore and PolicyRecord from their respective modules
    from metta.agent.policy_metadata import PolicyMetadata
    from metta.agent.policy_record import PolicyRecord
    from metta.agent.policy_store import PolicyStore

    # Create minimal config
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "run": "test_run",
            "run_dir": tempfile.mkdtemp(),
            "trainer": {
                "checkpoint": {"checkpoint_dir": tempfile.mkdtemp()},
                "num_workers": 1,
            },
            "data_dir": tempfile.mkdtemp(),
            # Add minimal agent config needed for make_policy
            "agent": {
                "type": "metta",
                "hidden_size": 256,
                "num_layers": 2,
            },
        }
    )

    # Create PolicyStore (without wandb_run for simplicity)
    policy_store = PolicyStore(cfg, wandb_run=None)

    # Create a test MettaAgent directly (simulating what make_policy would do)
    # For testing, we'll use MinimalPolicy but save it as if it's a MettaAgent
    policy = MinimalPolicy()

    # Create metadata using PolicyMetadata class
    metadata = PolicyMetadata(
        action_names=["move", "turn"],
        agent_step=100,
        epoch=5,
        generation=1,
        train_time=60.0,
    )

    # Save the policy
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        print(temp_path)

        # create a policy
        pr = policy_store.create_empty_policy_record(name=temp_path)
        pr.metadata = metadata
        pr.policy = policy

        # Save
        policy_store.save(pr)

        # Test that the save format is correct (without loading back)
        # This tests the critical part - that we don't have pydantic issues
        checkpoint = torch.load(temp_path, map_location="cpu", weights_only=False)

        # With the old simple approach, we save the entire PolicyRecord
        assert isinstance(checkpoint, PolicyRecord)

        # Verify metadata is properly saved
        assert checkpoint.metadata["action_names"] == ["move", "turn"]
        assert checkpoint.metadata["agent_step"] == 100
        assert checkpoint.metadata["epoch"] == 5
        assert checkpoint.metadata["generation"] == 1
        assert checkpoint.metadata["train_time"] == 60.0

        print("✅ Checkpoint format verified - using simple torch.save approach!")

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_policy_record_backwards_compatibility():
    """Test that PolicyRecord can handle old metadata attribute names"""
    from metta.agent.policy_metadata import PolicyMetadata
    from metta.agent.policy_record import PolicyRecord
    from metta.agent.policy_store import PolicyStore

    # Create minimal config for PolicyStore
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "run": "test_run",
            "run_dir": tempfile.mkdtemp(),
            "trainer": {
                "checkpoint": {"checkpoint_dir": tempfile.mkdtemp()},
                "num_workers": 1,
            },
            "data_dir": tempfile.mkdtemp(),
        }
    )

    policy_store = PolicyStore(cfg, wandb_run=None)

    # Test different old attribute names
    old_attribute_names = ["checkpoint", "meta_data", "_meta_data", "_checkpoint"]

    for old_name in old_attribute_names:
        # Create a PolicyRecord without using the normal constructor
        # to simulate loading an old checkpoint
        pr = PolicyRecord.__new__(PolicyRecord)
        pr._policy_store = policy_store
        pr.name = "test_policy"
        pr.uri = "file:///tmp/test.pt"
        pr._cached_policy = None

        # Set metadata using the old attribute name
        old_metadata = {
            "action_names": ["move", "turn"],
            "agent_step": 100,
            "epoch": 5,
            "generation": 1,
            "train_time": 60.0,
        }
        setattr(pr, old_name, old_metadata)

        # Access metadata property - this should trigger backwards compatibility
        metadata = pr.metadata

        # Verify it was converted properly
        assert isinstance(metadata, PolicyMetadata)
        assert metadata["action_names"] == ["move", "turn"]
        assert metadata["agent_step"] == 100
        assert metadata["epoch"] == 5
        assert metadata["generation"] == 1
        assert metadata["train_time"] == 60.0

        # Verify the old attribute was not removed (for safety)
        assert hasattr(pr, old_name)

        # Verify the new _metadata attribute exists
        assert hasattr(pr, "_metadata")

        print(f"✅ Backwards compatibility test passed for old attribute name: {old_name}")

    # Test with no metadata attribute at all
    pr_no_metadata = PolicyRecord.__new__(PolicyRecord)
    pr_no_metadata._policy_store = policy_store
    pr_no_metadata.name = "test_policy"
    pr_no_metadata.uri = "file:///tmp/test.pt"
    pr_no_metadata._cached_policy = None

    # This should raise AttributeError
    try:
        _ = pr_no_metadata.metadata
        assert False, "Expected AttributeError when no metadata found"
    except AttributeError as e:
        assert "No metadata found" in str(e)
        print("✅ Correctly raised AttributeError when no metadata found")


if __name__ == "__main__":
    test_policy_save_load_without_pydantic()
    test_policy_record_backwards_compatibility()
    print("✅ All tests passed!")
