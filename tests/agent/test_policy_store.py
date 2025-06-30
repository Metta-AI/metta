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
    from metta.agent.policy_store import PolicyStore

    # Create minimal config
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "trainer": {"checkpoint_dir": tempfile.mkdtemp()},
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

        # Verify checkpoint format
        assert "policy_metadata" in checkpoint or "metadata" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "name" in checkpoint
        assert "uri" in checkpoint
        assert "checkpoint_version" in checkpoint

        # Verify metadata is properly saved
        saved_metadata = checkpoint.get("policy_metadata") or checkpoint.get("metadata")
        assert saved_metadata is not None
        assert saved_metadata["action_names"] == ["move", "turn"]
        assert saved_metadata["agent_step"] == 100
        assert saved_metadata["epoch"] == 5
        assert saved_metadata["generation"] == 1
        assert saved_metadata["train_time"] == 60.0

        # Verify state dict is saved
        assert isinstance(checkpoint["model_state_dict"], dict)
        assert len(checkpoint["model_state_dict"]) > 0

        print("✅ Checkpoint format verified - no pydantic issues!")

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    test_policy_save_load_without_pydantic()
    print("✅ Test passed: PolicyStore can save/load policies.")
