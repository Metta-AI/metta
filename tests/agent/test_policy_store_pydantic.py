#!/usr/bin/env python3
"""Test that PolicyStore can save/load policies without pydantic dependency"""

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
    from metta.agent.policy_store import PolicyStore

    # Create minimal config
    cfg = OmegaConf.create(
        {
            "device": "cpu",
            "trainer": {"checkpoint_dir": tempfile.mkdtemp()},
            "data_dir": tempfile.mkdtemp(),
        }
    )

    # Create PolicyStore (without wandb_run for simplicity)
    policy_store = PolicyStore(cfg, wandb_run=None)

    # Create a test policy
    policy = MinimalPolicy()

    # Create metadata without pydantic objects
    metadata = {
        "action_names": ["move", "turn"],
        "agent_step": 100,
        "epoch": 5,
        "generation": 1,
        "train_time": 60.0,
    }

    # Save the policy
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        # Save using the convenience method
        policy_store.save("test_policy", temp_path, policy, metadata)

        # Try to load it back
        loaded_pr = policy_store.load_from_uri(f"file://{temp_path}")
        loaded_policy = loaded_pr.policy()

        # Verify the loaded policy works
        test_input = torch.randn(1, 10)
        output = loaded_policy(test_input)

        # Assertions
        # The loaded policy may not be the exact same class due to module externing
        assert type(loaded_policy).__name__ == "MinimalPolicy"
        assert loaded_pr.metadata == metadata
        assert output.shape == torch.Size([1, 10])

        # Verify the loaded policy has the expected structure
        assert hasattr(loaded_policy, "fc")
        assert hasattr(loaded_policy, "components")
        assert "_core_" in loaded_policy.components
        assert "_value_" in loaded_policy.components
        assert "_action_" in loaded_policy.components

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    test_policy_save_load_without_pydantic()
    print("✅ Test passed: PolicyStore can save/load without pydantic dependency.")
