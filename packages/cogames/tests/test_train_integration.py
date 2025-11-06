"""Integration tests for training and evaluation with different policies."""

import pathlib
import shutil
import tempfile

import pytest
import torch

import cogames.cli.mission
import cogames.train


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp(prefix="cogames_test_")
    yield pathlib.Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_env_config():
    """Get a small test game configuration."""
    return cogames.cli.mission.get_mission("machina_1")[1]


@pytest.mark.timeout(120)
def test_train_lstm_policy(test_env_config, temp_checkpoint_dir):
    """Test training with LSTMPolicy for 1000 steps."""
    cogames.train.train(
        env_cfg=test_env_config,
        policy_class_path="mettagrid.policy.lstm.LSTMPolicy",
        device=torch.device("cpu"),
        initial_weights_path=None,
        num_steps=1000,
        checkpoints_path=temp_checkpoint_dir,
        seed=42,
        batch_size=64,
        minibatch_size=64,
        vector_num_envs=1,
        vector_batch_size=1,
        vector_num_workers=1,
    )

    # Check that checkpoints were created
    checkpoints = list(temp_checkpoint_dir.rglob("*.pt"))
    assert len(checkpoints) > 0, f"Should have at least one checkpoint in {temp_checkpoint_dir}"

    # Verify checkpoint can be loaded
    checkpoint = checkpoints[0]
    state_dict = torch.load(checkpoint, map_location="cpu")
    assert isinstance(state_dict, dict), "Checkpoint should be a state dict"


# RandomPolicy is not trainable - it doesn't implement TrainablePolicy interface
# so we skip testing it with the train function
@pytest.mark.timeout(180)
def test_train_lstm_and_load_policy_data(test_env_config, temp_checkpoint_dir):
    """Test training LSTM policy, then loading it for evaluation."""
    import mettagrid.policy.lstm

    # Train the policy
    cogames.train.train(
        env_cfg=test_env_config,
        policy_class_path="mettagrid.policy.lstm.LSTMPolicy",
        device=torch.device("cpu"),
        initial_weights_path=None,
        num_steps=1000,
        checkpoints_path=temp_checkpoint_dir,
        seed=42,
        batch_size=64,
        minibatch_size=64,
        vector_num_envs=1,
        vector_batch_size=1,
        vector_num_workers=1,
    )

    # Find the saved checkpoint
    checkpoints = list(temp_checkpoint_dir.rglob("*.pt"))
    assert len(checkpoints) > 0, f"Should have at least one checkpoint in {temp_checkpoint_dir}"

    # Load the checkpoint into a new policy
    import mettagrid.policy.policy_env_interface

    policy_env_info = mettagrid.policy.policy_env_interface.PolicyEnvInterface.from_mg_cfg(test_env_config)
    policy = mettagrid.policy.lstm.LSTMPolicy(policy_env_info)
    policy.load_policy_data(str(checkpoints[0]))

    # Verify the policy network was loaded successfully
    import torch.nn as nn

    assert isinstance(policy.network(), nn.Module)
    # Verify the network has parameters (was loaded)
    assert sum(p.numel() for p in policy.network().parameters()) > 0
