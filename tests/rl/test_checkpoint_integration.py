"""
Integration tests for the checkpoint system with real agents and environments.
Tests the full pipeline of training, saving, loading, and using checkpoints.
"""

import tempfile
from pathlib import Path

import pytest
import torch

import metta.mettagrid.config.envs as eb
from metta.agent.agent_config import AgentConfig
from metta.agent.metta_agent import MettaAgent
from metta.agent.utils import obs_to_td
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.checkpoint_manager import CheckpointManager, get_checkpoint_uri_from_dir
from metta.rl.policy_management import resolve_policy
from metta.rl.system_config import SystemConfig


@pytest.fixture
def temp_run_dir():
    """Create a temporary directory for checkpoint tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def create_env_and_agent():
    """Create a real environment and agent for testing."""
    # Create a small test environment
    env_config = eb.make_navigation(num_agents=1)
    env_config.game.max_steps = 100
    env_config.game.map_builder.width = 8
    env_config.game.map_builder.height = 8

    # Create environment
    env = MettaGridEnv(env_config, render_mode=None)

    # Create system and agent configs
    system_cfg = SystemConfig(device="cpu")
    agent_cfg = AgentConfig(name="fast")

    # Create the agent
    agent = MettaAgent(
        env=env,
        system_cfg=system_cfg,
        policy_architecture_cfg=agent_cfg,
    )

    # Initialize agent to environment
    features = env.get_observation_features()
    agent.initialize_to_environment(features, env.action_names, env.max_action_args, device="cpu")

    return env, agent


class TestCheckpointIntegration:
    """Test checkpoint system with real agents and environments."""

    def test_save_and_load_real_agent(self, temp_run_dir, create_env_and_agent):
        """Test saving and loading a real MettaAgent through checkpoints."""
        env, agent = create_env_and_agent

        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(run_dir=temp_run_dir, run_name="test_run")

        # Run agent to generate some output before saving
        obs, _ = env.reset()
        output_before = agent(obs_to_td(obs))
        values_before = output_before["values"].clone()

        # Save the agent with metadata
        metadata = {
            "agent_step": 1000,
            "total_time": 60.5,
            "score": 0.85,  # Add a score for testing
        }
        checkpoint_manager.save_agent(agent, epoch=1, metadata=metadata)

        # Verify checkpoint was created with correct filename format
        checkpoint_dir = Path(temp_run_dir) / "test_run" / "checkpoints"
        expected_file = checkpoint_dir / "test_run.e1.s1000.t60.sc8500.pt"
        assert expected_file.exists()

        # Load the agent back
        loaded_agent = checkpoint_manager.load_agent(epoch=1)

        # Verify loaded agent works
        agent.eval()
        loaded_agent.eval()
        output_after = loaded_agent(obs_to_td(obs))

        # Values should be identical when in eval mode
        torch.testing.assert_close(values_before, output_after["values"])

    def test_uri_based_loading(self, temp_run_dir, create_env_and_agent):
        """Test loading agents through URI system."""
        env, agent = create_env_and_agent

        # Save agent using checkpoint manager
        checkpoint_manager = CheckpointManager(run_dir=temp_run_dir, run_name="uri_test")
        metadata = {"agent_step": 5000, "total_time": 120.0}
        checkpoint_manager.save_agent(agent, epoch=3, metadata=metadata)

        # Get URI to the checkpoint directory
        checkpoint_dir = Path(temp_run_dir) / "uri_test" / "checkpoints"
        dir_uri = f"file://{checkpoint_dir}"

        # Load using resolve_policy with directory URI
        loaded_policy = resolve_policy(dir_uri, device="cpu")

        # Verify it's a valid agent
        assert callable(loaded_policy)

        # Test with direct file URI
        file_uri = get_checkpoint_uri_from_dir(str(checkpoint_dir))
        loaded_from_file = resolve_policy(file_uri, device="cpu")

        # Both should work
        obs, _ = env.reset()
        output1 = loaded_policy(obs_to_td(obs))
        output2 = loaded_from_file(obs_to_td(obs))

        assert "actions" in output1
        assert "actions" in output2

    def test_checkpoint_with_training_progress(self, temp_run_dir, create_env_and_agent):
        """Test saving multiple checkpoints during training progression."""
        env, agent = create_env_and_agent
        checkpoint_manager = CheckpointManager(run_dir=temp_run_dir, run_name="training_run")

        # Simulate training loop with periodic saves
        obs, _ = env.reset()
        agent_step = 0
        total_time = 0.0

        for epoch in range(1, 4):
            # Simulate some training steps
            for _step in range(100):
                output = agent(obs_to_td(obs))
                actions = output["actions"].numpy()
                obs, rewards, terminated, truncated, _ = env.step(actions)

                if terminated.any() or truncated.any():
                    obs, _ = env.reset()

                agent_step += 1
                total_time += 0.01  # Simulate time passing

            # Save checkpoint at end of epoch
            score = 0.5 + epoch * 0.1  # Simulate improving scores
            metadata = {"agent_step": agent_step, "total_time": total_time, "score": score}
            checkpoint_manager.save_agent(agent, epoch=epoch, metadata=metadata)

        # Verify we have 3 checkpoints
        checkpoint_dir = Path(temp_run_dir) / "training_run" / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) == 3

        # Load the best checkpoint (should be epoch 3 with score 0.8)
        best_checkpoint = checkpoint_manager.find_best_checkpoint(metric="score")
        assert best_checkpoint is not None

        # Parse the best checkpoint to verify it's epoch 3
        from metta.rl.checkpoint_manager import parse_checkpoint_filename

        _, epoch, _, _, score = parse_checkpoint_filename(Path(best_checkpoint).name)
        assert epoch == 3
        assert abs(score - 0.8) < 0.001  # Account for float precision

    def test_multi_agent_checkpoint(self, temp_run_dir):
        """Test checkpointing with multi-agent environments."""
        # Create multi-agent environment
        env_config = eb.make_arena(num_agents=4)
        env_config.game.max_steps = 50
        env_config.game.map_builder.width = 12
        env_config.game.map_builder.height = 12

        env = MettaGridEnv(env_config, render_mode=None)

        # Create agent with attention architecture for multi-agent
        system_cfg = SystemConfig(device="cpu")
        agent_cfg = AgentConfig(name="latent_attn_tiny")

        agent = MettaAgent(
            env=env,
            system_cfg=system_cfg,
            policy_architecture_cfg=agent_cfg,
        )

        # Initialize
        features = env.get_observation_features()
        agent.initialize_to_environment(features, env.action_names, env.max_action_args, device="cpu")

        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(run_dir=temp_run_dir, run_name="multi_agent")

        # Run a few steps
        obs, _ = env.reset()
        for _ in range(10):
            output = agent(obs_to_td(obs))
            actions = output["actions"].numpy()
            obs, _, terminated, truncated, _ = env.step(actions)
            if terminated.any() or truncated.any():
                obs, _ = env.reset()

        # Save checkpoint
        metadata = {"agent_step": 10, "total_time": 1.0}
        checkpoint_manager.save_agent(agent, epoch=1, metadata=metadata)

        # Load and verify
        loaded_agent = checkpoint_manager.load_agent(epoch=1)

        # Test loaded agent with multi-agent batch
        obs, _ = env.reset()
        output = loaded_agent(obs_to_td(obs))

        # Should handle all 4 agents
        assert output["actions"].shape[0] == 4
        assert output["values"].shape[0] == 4

    def test_checkpoint_cleanup(self, temp_run_dir, create_env_and_agent):
        """Test automatic cleanup of old checkpoints."""
        env, agent = create_env_and_agent
        checkpoint_manager = CheckpointManager(run_dir=temp_run_dir, run_name="cleanup_test")

        # Save multiple checkpoints
        for epoch in range(1, 8):
            metadata = {
                "agent_step": epoch * 1000,
                "total_time": epoch * 10.0,
            }
            checkpoint_manager.save_agent(agent, epoch=epoch, metadata=metadata)

        # Clean up old checkpoints, keeping only last 3
        checkpoint_manager.cleanup_old_checkpoints(keep_last_n=3)

        # Verify only 3 checkpoints remain
        checkpoint_dir = Path(temp_run_dir) / "cleanup_test" / "checkpoints"
        remaining = list(checkpoint_dir.glob("*.pt"))
        assert len(remaining) == 3

        # Verify these are the latest 3 (epochs 5, 6, 7)
        from metta.rl.checkpoint_manager import parse_checkpoint_filename

        epochs = sorted([parse_checkpoint_filename(f.name)[1] for f in remaining])
        assert epochs == [5, 6, 7]

    def test_trainer_state_integration(self, temp_run_dir, create_env_and_agent):
        """Test saving and loading trainer state alongside agent."""
        env, agent = create_env_and_agent
        checkpoint_manager = CheckpointManager(run_dir=temp_run_dir, run_name="trainer_test")

        # Create a simple optimizer for testing
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)

        # Save agent
        metadata = {"agent_step": 2000, "total_time": 45.0}
        checkpoint_manager.save_agent(agent, epoch=2, metadata=metadata)

        # Save trainer state
        trainer_state = {
            "epoch": 2,
            "agent_step": 2000,
            "optimizer_state_dict": optimizer.state_dict(),
            "training_metrics": {"loss": 0.5, "value_loss": 0.3},
        }
        checkpoint_manager.save_trainer_state(trainer_state, epoch=2)

        # Load trainer state back
        loaded_state = checkpoint_manager.load_trainer_state(epoch=2)

        # Verify state was preserved
        assert loaded_state["epoch"] == 2
        assert loaded_state["agent_step"] == 2000
        assert "optimizer_state_dict" in loaded_state
        assert loaded_state["training_metrics"]["loss"] == 0.5

    def test_different_architectures(self, temp_run_dir):
        """Test checkpointing with different agent architectures."""
        architectures = ["fast", "latent_attn_tiny"]

        for arch_name in architectures:
            # Create environment
            env_config = eb.make_navigation(num_agents=1)
            env = MettaGridEnv(env_config, render_mode=None)

            # Create agent with specific architecture
            system_cfg = SystemConfig(device="cpu")
            agent_cfg = AgentConfig(name=arch_name)

            agent = MettaAgent(
                env=env,
                system_cfg=system_cfg,
                policy_architecture_cfg=agent_cfg,
            )

            # Initialize
            features = env.get_observation_features()
            agent.initialize_to_environment(features, env.action_names, env.max_action_args, device="cpu")

            # Create checkpoint manager
            run_name = f"arch_{arch_name}"
            checkpoint_manager = CheckpointManager(run_dir=temp_run_dir, run_name=run_name)

            # Save and load
            metadata = {"agent_step": 100, "total_time": 10.0}
            checkpoint_manager.save_agent(agent, epoch=1, metadata=metadata)

            loaded_agent = checkpoint_manager.load_agent(epoch=1)

            # Test forward pass
            obs, _ = env.reset()
            output = loaded_agent(obs_to_td(obs))

            # All architectures should work after loading
            assert "actions" in output
            assert "values" in output
            assert output["actions"].shape == (obs.shape[0], 2)

    def test_score_based_selection(self, temp_run_dir, create_env_and_agent):
        """Test selecting checkpoints based on score metric."""
        env, agent = create_env_and_agent
        checkpoint_manager = CheckpointManager(run_dir=temp_run_dir, run_name="score_test")

        # Save checkpoints with varying scores
        scores = [0.3, 0.7, 0.5, 0.9, 0.6]
        for epoch, score in enumerate(scores, 1):
            metadata = {"agent_step": epoch * 1000, "total_time": epoch * 10.0, "score": score}
            checkpoint_manager.save_agent(agent, epoch=epoch, metadata=metadata)

        # Select top 2 checkpoints by score
        top_checkpoints = checkpoint_manager.select_checkpoints(metric="score", top_k=2, descending=True)

        assert len(top_checkpoints) == 2

        # Parse and verify these are the highest scoring
        from metta.rl.checkpoint_manager import parse_checkpoint_filename

        selected_scores = []
        for checkpoint_path in top_checkpoints:
            _, _, _, _, score = parse_checkpoint_filename(Path(checkpoint_path).name)
            selected_scores.append(score)

        # Should have selected 0.9 and 0.7
        assert sorted(selected_scores, reverse=True) == [0.9, 0.7]

    def test_checkpoint_exists_check(self, temp_run_dir, create_env_and_agent):
        """Test checking if checkpoints exist."""
        env, agent = create_env_and_agent
        checkpoint_manager = CheckpointManager(run_dir=temp_run_dir, run_name="exists_test")

        # Initially no checkpoints
        assert not checkpoint_manager.exists()

        # Save a checkpoint
        metadata = {"agent_step": 500, "total_time": 5.0}
        checkpoint_manager.save_agent(agent, epoch=1, metadata=metadata)

        # Now checkpoints exist
        assert checkpoint_manager.exists()

        # Check latest epoch
        assert checkpoint_manager.get_latest_epoch() == 1

        # Save another
        metadata = {"agent_step": 1000, "total_time": 10.0}
        checkpoint_manager.save_agent(agent, epoch=2, metadata=metadata)

        # Latest should be 2
        assert checkpoint_manager.get_latest_epoch() == 2
