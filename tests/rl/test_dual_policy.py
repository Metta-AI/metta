"""Tests for the dual-policy training system."""

from unittest.mock import Mock

import pytest
import torch

from metta.rl.scripted_npc import ScriptedNPC, create_scripted_npc
from metta.rl.trainer_config import DualPolicyConfig, ScriptedNPCConfig
from metta.rl.util.dual_policy_rollout import DualPolicyRollout


class TestScriptedNPC:
    """Test scripted NPC functionality."""

    def test_roomba_npc_creation(self):
        """Test creating a roomba NPC."""
        config = ScriptedNPCConfig(type="roomba", roomba_direction="clockwise")
        num_agents = 4
        device = torch.device("cpu")

        npc = create_scripted_npc(config, num_agents, device)
        assert isinstance(npc, ScriptedNPC)
        assert npc.config.type == "roomba"
        assert len(npc.agent_states) == num_agents

    def test_grid_search_npc_creation(self):
        """Test creating a grid search NPC."""
        config = ScriptedNPCConfig(type="grid_search", grid_search_pattern="spiral")
        num_agents = 4
        device = torch.device("cpu")

        npc = create_scripted_npc(config, num_agents, device)
        assert isinstance(npc, ScriptedNPC)
        assert npc.config.type == "grid_search"
        assert len(npc.agent_states) == num_agents

    def test_npc_action_generation(self):
        """Test that NPCs can generate actions."""
        config = ScriptedNPCConfig(type="roomba")
        num_agents = 4
        device = torch.device("cpu")

        npc = create_scripted_npc(config, num_agents, device)

        # Create dummy observations
        observations = torch.randn(num_agents, 11, 11, 24)  # Example observation shape

        actions = npc.get_actions(observations)
        assert actions.shape == (num_agents, 2)
        assert actions.dtype == torch.int32


class TestDualPolicyConfig:
    """Test dual policy configuration."""

    def test_default_config(self):
        """Test default dual policy configuration."""
        config = DualPolicyConfig()
        assert not config.enabled
        assert config.policy_a_percentage == 0.5
        assert config.npc_type == "scripted"
        assert config.npc_policy_uri is None

    def test_enabled_config(self):
        """Test enabled dual policy configuration."""
        config = DualPolicyConfig(enabled=True, policy_a_percentage=0.7, npc_type="scripted")
        assert config.enabled
        assert config.policy_a_percentage == 0.7
        assert config.npc_type == "scripted"

    def test_checkpoint_config_validation(self):
        """Test that checkpoint config requires URI."""
        with pytest.raises(ValueError, match="npc_policy_uri must be set"):
            DualPolicyConfig(enabled=True, npc_type="checkpoint", npc_policy_uri=None)

    def test_valid_checkpoint_config(self):
        """Test valid checkpoint configuration."""
        config = DualPolicyConfig(enabled=True, npc_type="checkpoint", npc_policy_uri="path/to/checkpoint")
        assert config.enabled
        assert config.npc_type == "checkpoint"
        assert config.npc_policy_uri == "path/to/checkpoint"


class TestDualPolicyRollout:
    """Test dual policy rollout functionality."""

    def test_disabled_rollout(self):
        """Test that disabled rollout falls back to single policy."""
        config = DualPolicyConfig(enabled=False)
        mock_policy_store = Mock()
        num_agents = 4
        device = torch.device("cpu")

        rollout = DualPolicyRollout(config, mock_policy_store, num_agents, device)
        assert rollout.config.enabled is False
        assert rollout.dual_policy_rollout is None

    def test_scripted_npc_rollout_creation(self):
        """Test creating rollout with scripted NPCs."""
        config = DualPolicyConfig(enabled=True, policy_a_percentage=0.5, npc_type="scripted")
        mock_policy_store = Mock()
        num_agents = 4
        device = torch.device("cpu")

        rollout = DualPolicyRollout(config, mock_policy_store, num_agents, device)
        assert rollout.config.enabled
        assert rollout.policy_a_count == 2
        assert rollout.npc_count == 2
        assert rollout.npc_policy is not None

    def test_agent_assignment_masks(self):
        """Test that agent assignment masks are correct."""
        config = DualPolicyConfig(
            enabled=True,
            policy_a_percentage=0.6,  # 60% policy A, 40% NPC
        )
        mock_policy_store = Mock()
        num_agents = 10
        device = torch.device("cpu")

        rollout = DualPolicyRollout(config, mock_policy_store, num_agents, device)

        # Check agent counts
        assert rollout.policy_a_count == 6
        assert rollout.npc_count == 4

        # Check masks
        assert rollout.policy_a_mask.sum() == 6
        assert rollout.npc_mask.sum() == 4
        assert (rollout.policy_a_mask & rollout.npc_mask).sum() == 0  # No overlap

    def test_reward_tracking(self):
        """Test reward tracking functionality."""
        config = DualPolicyConfig(enabled=True, policy_a_percentage=0.5)
        mock_policy_store = Mock()
        num_agents = 4
        device = torch.device("cpu")

        rollout = DualPolicyRollout(config, mock_policy_store, num_agents, device)

        # Create dummy rewards
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)

        # Track rewards
        rollout.track_rewards(rewards)

        # Check that rewards were tracked
        assert len(rollout.policy_a_rewards) == 1
        assert len(rollout.npc_rewards) == 1
        assert len(rollout.combined_rewards) == 1

        # Get stats
        stats = rollout.get_reward_stats()
        assert "policy_a_reward" in stats
        assert "npc_reward" in stats
        assert "combined_reward" in stats

    def test_reward_tracking_reset(self):
        """Test that reward tracking can be reset."""
        config = DualPolicyConfig(enabled=True, policy_a_percentage=0.5)
        mock_policy_store = Mock()
        num_agents = 4
        device = torch.device("cpu")

        rollout = DualPolicyRollout(config, mock_policy_store, num_agents, device)

        # Add some rewards
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        rollout.track_rewards(rewards)

        # Reset
        rollout.reset_reward_tracking()

        # Check that rewards were cleared
        assert len(rollout.policy_a_rewards) == 0
        assert len(rollout.npc_rewards) == 0
        assert len(rollout.combined_rewards) == 0

    def test_agent_assignments_info(self):
        """Test getting agent assignment information."""
        config = DualPolicyConfig(enabled=True, policy_a_percentage=0.5, npc_type="scripted")
        mock_policy_store = Mock()
        num_agents = 4
        device = torch.device("cpu")

        rollout = DualPolicyRollout(config, mock_policy_store, num_agents, device)

        info = rollout.get_agent_assignments()
        assert info["policy_a_count"] == 2
        assert info["npc_count"] == 2
        assert info["policy_a_percentage"] == 0.5
        assert info["npc_type"] == "scripted"
        assert info["enabled"] is True


if __name__ == "__main__":
    pytest.main([__file__])
