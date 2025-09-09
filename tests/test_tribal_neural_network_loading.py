#!/usr/bin/env python3
"""
Test for loading and using trained neural networks in tribal environment.

This test verifies that we can:
1. Load a trained neural network from a checkpoint
2. Initialize the tribal environment
3. Run neural network inference on tribal observations
4. Validate the neural network outputs
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

# Add the project paths to sys.path for tribal bindings
sys.path.insert(0, str(Path(__file__).parent.parent / "tribal" / "bindings" / "generated"))

from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.tribal_genny import TribalEnvConfig, TribalGridEnv


@pytest.fixture
def sample_checkpoint_path():
    """Fixture to provide path to a sample checkpoint for testing."""
    checkpoint_dir = Path("./train_dir/test_tribal_neural_demo/test_tribal_neural_demo/checkpoints")
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))

    # Filter out trainer_state.pt
    model_files = [f for f in checkpoint_files if not f.name.startswith("trainer_state")]

    if not model_files:
        pytest.skip(
            "No trained tribal model found. Run training first: "
            "uv run ./tools/run.py experiments.recipes.tribal_basic.train --overrides run=test_tribal_neural_demo trainer.total_timesteps=1000"
        )

    return str(model_files[0])


@pytest.fixture
def tribal_environment():
    """Fixture to create a tribal environment for testing."""
    env_config = TribalEnvConfig(label="tribal_test", desync_episodes=True)
    env = TribalGridEnv(env_config)
    return env


class TestTribalNeuralNetworkLoading:
    """Test suite for tribal neural network loading and inference."""

    def test_checkpoint_loading(self, sample_checkpoint_path):
        """Test that we can successfully load a neural network checkpoint."""
        # Load the neural network
        policy = CheckpointManager.load_from_uri(f"file://{sample_checkpoint_path}")

        # Verify it's the correct type
        assert policy is not None
        assert hasattr(policy, "parameters"), "Policy should have parameters"

        # Check parameter count (should be around 565k for the fast agent)
        param_count = sum(p.numel() for p in policy.parameters())
        assert param_count > 500000, f"Expected >500k parameters, got {param_count}"
        assert param_count < 1000000, f"Expected <1M parameters, got {param_count}"

        print(f"✅ Loaded policy with {param_count:,} parameters")

    def test_environment_creation(self, tribal_environment):
        """Test that we can create and configure the tribal environment."""
        env = tribal_environment

        # Check basic properties
        assert hasattr(env, "num_agents"), "Environment should have num_agents property"
        assert hasattr(env, "action_names"), "Environment should have action_names property"

        # Verify expected values
        assert env.num_agents == 15, f"Expected 15 agents, got {env.num_agents}"
        expected_actions = ["NOOP", "MOVE", "ATTACK", "GET", "SWAP", "PUT"]
        assert env.action_names == expected_actions, f"Action names mismatch: {env.action_names}"

        print(f"✅ Environment created with {env.num_agents} agents and actions: {env.action_names}")

    def test_observation_structure(self, tribal_environment):
        """Test that environment observations have the expected structure."""
        env = tribal_environment

        # Reset environment and get observations
        observations, info = env.reset()

        # Check observation structure
        assert isinstance(observations, np.ndarray), "Observations should be numpy array"
        assert len(observations.shape) == 3, f"Expected 3D observations, got shape {observations.shape}"
        assert observations.shape[0] == 15, f"Expected 15 agents, got {observations.shape[0]}"
        assert observations.shape[1] == 200, f"Expected 200 tokens per agent, got {observations.shape[1]}"
        assert observations.shape[2] == 3, f"Expected 3 features per token, got {observations.shape[2]}"

        # Check data type and range
        assert observations.dtype == np.uint8, f"Expected uint8, got {observations.dtype}"
        assert observations.min() >= 0, "Observations should be non-negative"
        assert observations.max() <= 255, "Observations should be ≤255"

        print(f"✅ Observations shape: {observations.shape}, range: [{observations.min()}, {observations.max()}]")

    def test_neural_network_inference(self, sample_checkpoint_path, tribal_environment):
        """Test complete neural network inference pipeline."""
        # Load the neural network
        policy = CheckpointManager.load_from_uri(f"file://{sample_checkpoint_path}")
        policy.eval()  # Set to evaluation mode

        # Get observations from environment
        env = tribal_environment
        observations, info = env.reset()

        # Convert observations to tensor format
        obs_tensor = torch.from_numpy(observations).float()
        if len(obs_tensor.shape) == 3:
            obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension

        # Create TensorDict input with required keys
        batch_size = obs_tensor.shape[0]
        num_agents = obs_tensor.shape[1]
        flattened_obs = obs_tensor.reshape(batch_size * num_agents, *obs_tensor.shape[2:])

        input_td = TensorDict(
            {
                "env_obs": flattened_obs,
                "batch": torch.zeros(batch_size * num_agents, dtype=torch.long),
                "bptt": torch.zeros(batch_size * num_agents, dtype=torch.long),
            },
            batch_size=[batch_size * num_agents],
        )

        # Run neural network inference
        with torch.no_grad():
            output_td = policy(input_td)

        # Verify outputs exist and have reasonable properties
        # The actual output keys are '_action_', 'values', 'actions', etc.
        assert "_action_" in output_td or "values" in output_td, "Output should contain _action_ and/or values"

        if "_action_" in output_td:
            action_logits = output_td["_action_"]
            assert isinstance(action_logits, torch.Tensor), "Action logits should be tensor"
            assert action_logits.shape[0] == batch_size * num_agents, (
                f"Expected {batch_size * num_agents} agents, got {action_logits.shape[0]}"
            )
            assert action_logits.shape[1] > 0, "Action logits should have >0 actions"
            assert not torch.isnan(action_logits).any(), "Action logits should not contain NaN"
            print(
                f"✅ Action logits shape: {action_logits.shape}, range: [{action_logits.min():.3f}, {action_logits.max():.3f}]"
            )

        if "values" in output_td:
            values = output_td["values"]
            assert isinstance(values, torch.Tensor), "Values should be tensor"
            assert values.shape[0] == batch_size * num_agents, (
                f"Expected {batch_size * num_agents} agents, got {values.shape[0]}"
            )
            assert not torch.isnan(values).any(), "Values should not contain NaN"
            print(f"✅ Value estimates shape: {values.shape}, range: [{values.min():.3f}, {values.max():.3f}]")

        if "actions" in output_td:
            actions = output_td["actions"]
            assert isinstance(actions, torch.Tensor), "Actions should be tensor"
            assert actions.shape[0] == batch_size * num_agents, (
                f"Expected {batch_size * num_agents} agents, got {actions.shape[0]}"
            )
            print(f"✅ Sampled actions shape: {actions.shape}")

    def test_neural_network_architecture(self, sample_checkpoint_path):
        """Test that the neural network has the expected architecture components."""
        policy = CheckpointManager.load_from_uri(f"file://{sample_checkpoint_path}")

        # Check that it has the expected modular structure
        assert hasattr(policy, "policy"), "Policy should have policy attribute"
        policy_obj = policy.policy

        # Check for expected component names (from Fast agent architecture)
        expected_components = [
            "_obs_",
            "obs_normalizer",
            "cnn1",
            "cnn2",
            "obs_flattener",
            "fc1",
            "encoded_obs",
            "_core_",
            "critic_1",
            "_value_",
            "actor_1",
            "_action_embeds_",
            "actor_query",
            "_action_",
        ]

        if hasattr(policy_obj, "components"):
            actual_components = list(policy_obj.components.keys())
            for component in expected_components:
                assert component in actual_components, f"Missing component: {component}"
            print(f"✅ Neural network has all expected components: {len(actual_components)} total")

    def test_end_to_end_inference_consistency(self, sample_checkpoint_path, tribal_environment):
        """Test that multiple inference runs on the same input produce consistent results."""
        policy = CheckpointManager.load_from_uri(f"file://{sample_checkpoint_path}")
        policy.eval()

        env = tribal_environment
        observations, _ = env.reset()

        # Convert to input format
        obs_tensor = torch.from_numpy(observations).float().unsqueeze(0)
        batch_size, num_agents = obs_tensor.shape[:2]
        flattened_obs = obs_tensor.reshape(batch_size * num_agents, *obs_tensor.shape[2:])

        input_td = TensorDict(
            {
                "env_obs": flattened_obs,
                "batch": torch.zeros(batch_size * num_agents, dtype=torch.long),
                "bptt": torch.zeros(batch_size * num_agents, dtype=torch.long),
            },
            batch_size=[batch_size * num_agents],
        )

        # Run inference twice and check consistency
        with torch.no_grad():
            output1 = policy(input_td)
            output2 = policy(input_td)

        # Results should be identical (deterministic inference)
        if "_action_" in output1 and "_action_" in output2:
            torch.testing.assert_close(
                output1["_action_"], output2["_action_"], msg="Action logits should be deterministic"
            )

        if "values" in output1 and "values" in output2:
            torch.testing.assert_close(
                output1["values"], output2["values"], msg="Value estimates should be deterministic"
            )

        print("✅ Multiple inference runs produce consistent results")


if __name__ == "__main__":
    # Allow running as script for development
    pytest.main([__file__, "-v"])
