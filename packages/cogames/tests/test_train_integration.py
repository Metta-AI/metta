"""Integration tests for training and evaluation with different policies."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from cogames.cli.mission import get_mission
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
    return get_mission("machina_1")[1]


@pytest.mark.timeout(120)
def test_train_lstm_policy(test_env_config, temp_checkpoint_dir):
    """Test training with LSTMPolicy for 1000 steps."""
    train(
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


# RandomPolicy is not trainable - it returns None from network()
# so we skip testing it with the train function
@pytest.mark.timeout(180)
def test_train_lstm_and_load_policy_data(test_env_config, temp_checkpoint_dir):
    """Test training LSTM policy, then loading it for evaluation."""
    from mettagrid.policy.lstm import LSTMPolicy

    # Train the policy
    train(
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
    from mettagrid.policy.policy_env_interface import PolicyEnvInterface

    policy_env_info = PolicyEnvInterface.from_mg_cfg(test_env_config)
    policy = LSTMPolicy(policy_env_info)
    policy.load_policy_data(str(checkpoints[0]))

    # Verify the policy network was loaded successfully
    import torch.nn as nn

    assert isinstance(policy.network(), nn.Module)
    # Verify the network has parameters (was loaded)
    assert sum(p.numel() for p in policy.network().parameters()) > 0


cogames_root = Path(__file__).parent.parent


@pytest.mark.timeout(120)
def test_make_policy_trainable_and_train(temp_checkpoint_dir):
    """Test that make-policy --trainable generates a working trainable policy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        policy_file = tmpdir / "my_trainable_policy.py"

        # Generate the trainable policy template
        result = subprocess.run(
            ["uv", "run", "cogames", "make-policy", "--trainable", "-o", str(policy_file)],
            cwd=cogames_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"make-policy --trainable failed: {result.stderr}"
        assert policy_file.exists(), "Policy file was not created"

        # Add tmpdir to sys.path so the policy module can be imported
        sys.path.insert(0, str(tmpdir))
        try:
            # Train using the generated policy for a few steps
            train(
                env_cfg=get_mission("machina_1")[1],
                policy_class_path=f"{policy_file.stem}.MyTrainablePolicy",
                device=torch.device("cpu"),
                initial_weights_path=None,
                num_steps=100,
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
            assert len(checkpoints) > 0, "Training should produce checkpoints"
        finally:
            sys.path.remove(str(tmpdir))


@pytest.mark.timeout(60)
def test_make_policy_scripted_runs():
    """Test that make-policy --scripted generates a working scripted policy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        policy_file = tmpdir / "my_scripted_policy.py"

        # Generate the scripted policy template
        result = subprocess.run(
            ["uv", "run", "cogames", "make-policy", "--scripted", "-o", str(policy_file)],
            cwd=cogames_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"make-policy --scripted failed: {result.stderr}"
        assert policy_file.exists(), "Policy file was not created"

        # Verify the file contains expected content
        content = policy_file.read_text()
        assert "class StarterPolicy" in content
        assert "class StarterCogPolicyImpl" in content
        assert "def step_with_state" in content
        assert "MultiAgentPolicy" in content

        # Verify policy can be instantiated
        sys.path.insert(0, str(tmpdir))
        try:
            from mettagrid.policy.loader import initialize_or_load_policy
            from mettagrid.policy.policy import PolicySpec
            from mettagrid.policy.policy_env_interface import PolicyEnvInterface

            env_cfg = get_mission("machina_1")[1]
            policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
            policy_spec = PolicySpec(class_path=f"{policy_file.stem}.StarterPolicy")
            policy = initialize_or_load_policy(policy_env_info, policy_spec)

            # Verify it can create agent policies
            agent_policy = policy.agent_policy(0)
            assert agent_policy is not None
        finally:
            sys.path.remove(str(tmpdir))
