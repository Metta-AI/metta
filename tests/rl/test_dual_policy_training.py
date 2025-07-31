"""Tests for dual-policy training functionality."""

import pytest

from metta.rl.trainer_config import DualPolicyConfig, TrainerConfig


def test_dual_policy_config_validation():
    """Test that dual-policy config validation works correctly."""

    # Test valid config
    config_dict = {
        "enabled": True,
        "training_agents_pct": 0.5,
        "checkpoint_npc": {"uri": "wandb://test/uri", "type": "specific", "range": 1, "metric": "epoch"},
    }
    config = DualPolicyConfig(**config_dict)
    assert config.enabled is True
    assert config.training_agents_pct == 0.5
    assert config.checkpoint_npc.uri == "wandb://test/uri"

    # Test invalid config - enabled but no URI
    with pytest.raises(ValueError, match="checkpoint_npc.uri must be set when dual_policy.enabled is True"):
        DualPolicyConfig(enabled=True, training_agents_pct=0.5)

    # Test valid disabled config
    config = DualPolicyConfig(enabled=False, training_agents_pct=0.5)
    assert config.enabled is False


def test_trainer_config_with_dual_policy():
    """Test that TrainerConfig properly includes dual-policy configuration."""

    config_dict = {
        "num_workers": 4,
        "total_timesteps": 1000000,
        "ppo": {},
        "optimizer": {},
        "prioritized_experience_replay": {},
        "vtrace": {},
        "dual_policy": {
            "enabled": True,
            "training_agents_pct": 0.5,
            "checkpoint_npc": {"uri": "wandb://test/uri", "type": "specific", "range": 1, "metric": "epoch"},
        },
        "checkpoint": {"checkpoint_dir": "/tmp"},
        "simulation": {"replay_dir": "/tmp"},
        "kickstart": {},
        "hyperparameter_scheduler": {},
        "profiler": {"profile_dir": "/tmp"},
    }

    config = TrainerConfig(**config_dict)
    assert config.dual_policy.enabled is True
    assert config.dual_policy.training_agents_pct == 0.5
    assert config.dual_policy.checkpoint_npc.uri == "wandb://test/uri"


def test_dual_policy_config_defaults():
    """Test that dual-policy config has correct defaults."""

    config = DualPolicyConfig()
    assert config.enabled is False
    assert config.training_agents_pct == 0.5
    assert config.checkpoint_npc.uri is None


def test_trainer_config_dual_policy_defaults():
    """Test that TrainerConfig has dual-policy disabled by default."""

    config_dict = {
        "num_workers": 4,
        "total_timesteps": 1000000,
        "ppo": {},
        "optimizer": {},
        "prioritized_experience_replay": {},
        "vtrace": {},
        "dual_policy": {},
        "checkpoint": {"checkpoint_dir": "/tmp"},
        "simulation": {"replay_dir": "/tmp"},
        "kickstart": {},
        "hyperparameter_scheduler": {},
        "profiler": {"profile_dir": "/tmp"},
    }

    config = TrainerConfig(**config_dict)
    assert config.dual_policy.enabled is False
    assert config.dual_policy.training_agents_pct == 0.5
