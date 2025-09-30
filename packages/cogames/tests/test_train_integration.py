"""Integration tests for training and evaluation with different policies."""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from cogames import game
from cogames.train import train


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp(prefix="cogames_test_")
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_env_config():
    """Get a small test game configuration."""
    return game.get_game("machina_1")


@pytest.mark.timeout(120)  # 2 minute timeout
def test_train_simple_policy(test_env_config, temp_checkpoint_dir):
    """Test training with SimplePolicy for 1000 steps."""
    train(
        env_cfg=test_env_config,
        policy_class_path="cogames.policy.simple.SimplePolicy",
        device=torch.device("cpu"),
        initial_weights_path=None,
        num_steps=1000,
        checkpoints_path=temp_checkpoint_dir,
        seed=42,
        batch_size=256,
        minibatch_size=256,
    )

    # Check that checkpoints were created
    # PufferLib saves to {data_dir}/{run_id}/model_{epoch}.pt
    # Find any subdirectories with checkpoint files
    checkpoints = list(temp_checkpoint_dir.rglob("*.pt"))
    assert len(checkpoints) > 0, f"Should have at least one checkpoint in {temp_checkpoint_dir}"

    # Verify checkpoint can be loaded
    checkpoint = checkpoints[0]
    state_dict = torch.load(checkpoint, map_location="cpu")
    assert isinstance(state_dict, dict), "Checkpoint should be a state dict"


@pytest.mark.timeout(120)
def test_train_lstm_policy(test_env_config, temp_checkpoint_dir):
    """Test training with LSTMPolicy for 1000 steps."""
    train(
        env_cfg=test_env_config,
        policy_class_path="cogames.policy.lstm.LSTMPolicy",
        device=torch.device("cpu"),
        initial_weights_path=None,
        num_steps=1000,
        checkpoints_path=temp_checkpoint_dir,
        seed=42,
        batch_size=256,
        minibatch_size=256,
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


@pytest.mark.timeout(180)  # 3 minute timeout
def test_train_and_load_policy_data(test_env_config, temp_checkpoint_dir):
    """Test training a policy, then loading it for evaluation."""
    from cogames.policy.simple import SimplePolicy
    from mettagrid import MettaGridEnv

    # Train the policy
    train(
        env_cfg=test_env_config,
        policy_class_path="cogames.policy.simple.SimplePolicy",
        device=torch.device("cpu"),
        initial_weights_path=None,
        num_steps=1000,
        checkpoints_path=temp_checkpoint_dir,
        seed=42,
        batch_size=256,
        minibatch_size=256,
    )

    # Find the saved checkpoint
    checkpoints = list(temp_checkpoint_dir.rglob("*.pt"))
    assert len(checkpoints) > 0, f"Should have at least one checkpoint in {temp_checkpoint_dir}"

    # Load the checkpoint into a new policy
    env = MettaGridEnv(env_cfg=test_env_config)
    policy = SimplePolicy(env, torch.device("cpu"))
    policy.load_policy_data(str(checkpoints[0]))

    # Verify the policy can be used for inference
    obs, _ = env.reset()
    # Handle both dict-based (multi-agent) and array-based (vectorized) observations
    if isinstance(obs, dict):
        for agent_id, agent_obs in obs.items():
            agent_policy = policy.agent_policy(agent_id)
            action = agent_policy.step(agent_obs)
            assert action is not None
            assert len(action) == len(env.single_action_space.nvec)
    else:
        # Single observation - test with agent 0
        agent_policy = policy.agent_policy(0)
        action = agent_policy.step(obs)
        assert action is not None
        assert len(action) == len(env.single_action_space.nvec)


@pytest.mark.timeout(180)
def test_train_lstm_and_load_policy_data(test_env_config, temp_checkpoint_dir):
    """Test training LSTM policy, then loading it for evaluation."""
    from cogames.policy.lstm import LSTMPolicy
    from mettagrid import MettaGridEnv

    # Train the policy
    train(
        env_cfg=test_env_config,
        policy_class_path="cogames.policy.lstm.LSTMPolicy",
        device=torch.device("cpu"),
        initial_weights_path=None,
        num_steps=1000,
        checkpoints_path=temp_checkpoint_dir,
        seed=42,
        batch_size=256,
        minibatch_size=256,
    )

    # Find the saved checkpoint
    checkpoints = list(temp_checkpoint_dir.rglob("*.pt"))
    assert len(checkpoints) > 0, f"Should have at least one checkpoint in {temp_checkpoint_dir}"

    # Load the checkpoint into a new policy
    env = MettaGridEnv(env_cfg=test_env_config)
    policy = LSTMPolicy(env, torch.device("cpu"))
    policy.load_policy_data(str(checkpoints[0]))

    # Verify the policy can be used for inference with state
    obs, _ = env.reset()
    # Handle both dict-based (multi-agent) and array-based (vectorized) observations
    if isinstance(obs, dict):
        for agent_id, agent_obs in obs.items():
            agent_policy = policy.agent_policy(agent_id)
            # First step - no state
            action = agent_policy.step(agent_obs)
            assert action is not None
            assert len(action) == len(env.single_action_space.nvec)
    else:
        # Single observation - test with agent 0
        agent_policy = policy.agent_policy(0)
        # First step - no state
        action = agent_policy.step(obs)
        assert action is not None
        assert len(action) == len(env.single_action_space.nvec)
