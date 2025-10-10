"""Tests for latent dynamics policy configurations."""

from types import SimpleNamespace

import gymnasium as gym
import pytest

from metta.agent.policies.latent_dynamics import LatentDynamicsPolicyConfig, LatentDynamicsTinyConfig
from metta.rl.training import EnvironmentMetaData


@pytest.fixture
def env_metadata():
    """Create mock environment metadata."""
    action_names = ["move_north", "move_south", "move_east", "move_west", "attack", "gather", "craft", "wait"]
    feature_normalizations = {0: 1.0}

    obs_features = {
        "token_value": SimpleNamespace(id=0, normalization=1.0),
    }

    return EnvironmentMetaData(
        obs_width=11,
        obs_height=11,
        obs_features=obs_features,
        action_names=action_names,
        num_agents=1,
        observation_space=None,
        action_space=gym.spaces.Discrete(len(action_names)),
        feature_normalizations=feature_normalizations,
    )


def test_latent_dynamics_policy_config_creates_policy(env_metadata):
    """Test that LatentDynamicsPolicyConfig can create a policy."""
    config = LatentDynamicsPolicyConfig()
    policy = config.make_policy(env_metadata)

    assert policy is not None
    assert hasattr(policy, "forward")


def test_latent_dynamics_tiny_config_creates_policy(env_metadata):
    """Test that LatentDynamicsTinyConfig can create a policy."""
    config = LatentDynamicsTinyConfig()
    policy = config.make_policy(env_metadata)

    assert policy is not None
    assert hasattr(policy, "forward")


def test_latent_dynamics_policy_has_dynamics_component(env_metadata):
    """Test that the policy includes the latent dynamics component."""
    config = LatentDynamicsPolicyConfig()

    # Check that LatentDynamicsConfig is in components
    from metta.agent.components.dynamics import LatentDynamicsConfig

    dynamics_components = [c for c in config.components if isinstance(c, LatentDynamicsConfig)]

    assert len(dynamics_components) == 1, "Policy should have exactly one LatentDynamicsConfig"
    assert dynamics_components[0].name == "latent_dynamics"


def test_policy_config_parameters(env_metadata):
    """Test that policy configuration has expected parameters."""
    config = LatentDynamicsPolicyConfig()

    # Check internal dimensions
    assert config._latent_dim == 64
    assert config._core_out_dim == 32
    assert config._dynamics_latent_dim == 32

    # Check dynamics hyperparameters
    assert config._dynamics_encoder_hidden == [64, 64]
    assert config._dynamics_decoder_hidden == [64, 64]


def test_tiny_config_smaller_than_regular(env_metadata):
    """Test that tiny config has smaller dimensions than regular config."""
    regular = LatentDynamicsPolicyConfig()
    tiny = LatentDynamicsTinyConfig()

    assert tiny._latent_dim < regular._latent_dim
    assert tiny._core_out_dim < regular._core_out_dim
    assert tiny._dynamics_latent_dim < regular._dynamics_latent_dim
    assert len(tiny._dynamics_encoder_hidden) <= len(regular._dynamics_encoder_hidden)
