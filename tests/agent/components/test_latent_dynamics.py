"""Tests for latent-variable dynamics model component."""

import pytest
import torch
from tensordict import TensorDict

from metta.agent.components.dynamics import LatentDynamicsConfig, LatentDynamicsModelComponent


@pytest.fixture
def config():
    """Create a basic configuration for testing."""
    return LatentDynamicsConfig(
        name="test_latent_dynamics",
        in_key="obs",
        out_key="latent_hidden",
        action_key="actions",
        latent_dim=16,
        encoder_hidden=[32, 32],
        decoder_hidden=[32, 32],
        auxiliary_hidden=[16],
        beta_kl=0.01,
        gamma_auxiliary=1.0,
        future_horizon=3,
        future_type="returns",
        use_auxiliary=True,
    )


@pytest.fixture
def component(config):
    """Create a component instance."""
    return LatentDynamicsModelComponent(config=config, env=None)


def test_component_creation(config):
    """Test that the component can be created."""
    component = LatentDynamicsModelComponent(config=config, env=None)
    assert component is not None
    assert component._latent_dim == 16


def test_encode(component):
    """Test encoding a transition."""
    batch_size = 4
    obs_dim = 64
    action_dim = 8

    obs_t = torch.randn(batch_size, obs_dim)
    action_t = torch.randint(0, action_dim, (batch_size,))
    obs_next = torch.randn(batch_size, obs_dim)

    component._action_dim = action_dim

    z_mean, z_logvar = component.encode(obs_t, action_t, obs_next)

    assert z_mean.shape == (batch_size, component._latent_dim)
    assert z_logvar.shape == (batch_size, component._latent_dim)


def test_reparameterize(component):
    """Test reparameterization trick."""
    batch_size = 4
    latent_dim = component._latent_dim

    z_mean = torch.randn(batch_size, latent_dim)
    z_logvar = torch.randn(batch_size, latent_dim)

    z = component.reparameterize(z_mean, z_logvar)

    assert z.shape == (batch_size, latent_dim)

    # Test that sampling is stochastic
    z2 = component.reparameterize(z_mean, z_logvar)
    assert not torch.allclose(z, z2)


def test_decode(component):
    """Test decoding latent to predict next observation."""
    batch_size = 4
    obs_dim = 64
    action_dim = 8
    latent_dim = component._latent_dim

    obs_t = torch.randn(batch_size, obs_dim)
    action_t = torch.randint(0, action_dim, (batch_size,))
    z = torch.randn(batch_size, latent_dim)

    component._action_dim = action_dim

    obs_next_pred = component.decode(obs_t, action_t, z)

    # Output should match observation dimension
    assert obs_next_pred.shape[0] == batch_size


def test_predict_future(component):
    """Test auxiliary future prediction."""
    batch_size = 4
    latent_dim = component._latent_dim

    z = torch.randn(batch_size, latent_dim)

    future_pred = component.predict_future(z)

    assert future_pred is not None
    assert future_pred.shape[0] == batch_size


def test_forward_training_mode(component):
    """Test forward pass in training mode (with next observations)."""
    batch_size = 4
    obs_dim = 64
    action_dim = 8

    component._action_dim = action_dim

    td = TensorDict(
        {
            "obs": torch.randn(batch_size, obs_dim),
            "obs_next": torch.randn(batch_size, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size,)),
        },
        batch_size=batch_size,
    )

    output_td = component.forward(td)

    # Check all expected outputs are present
    assert "latent_mean" in output_td
    assert "latent_logvar" in output_td
    assert "latent" in output_td
    assert "obs_next_pred" in output_td
    assert "future_pred" in output_td
    assert "latent_hidden" in output_td

    # Check shapes
    assert output_td["latent"].shape == (batch_size, component._latent_dim)


def test_forward_inference_mode(component):
    """Test forward pass in inference mode (without next observations)."""
    batch_size = 4
    obs_dim = 64
    action_dim = 8

    component._action_dim = action_dim

    td = TensorDict(
        {
            "obs": torch.randn(batch_size, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size,)),
        },
        batch_size=batch_size,
    )

    output_td = component.forward(td)

    # In inference mode, should still produce predictions
    assert "latent" in output_td
    assert "obs_next_pred" in output_td
    assert "latent_hidden" in output_td

    # Should sample from prior (mean=0, var=1)
    assert output_td["latent"].shape == (batch_size, component._latent_dim)


def test_forward_without_actions(component):
    """Test forward pass with default zero actions."""
    batch_size = 4
    obs_dim = 64

    td = TensorDict(
        {
            "obs": torch.randn(batch_size, obs_dim),
        },
        batch_size=batch_size,
    )

    output_td = component.forward(td)

    # Should handle missing actions gracefully
    assert "latent" in output_td
    assert "obs_next_pred" in output_td


def test_backward_pass(component):
    """Test that gradients flow through the component."""
    batch_size = 4
    obs_dim = 64
    action_dim = 8

    component._action_dim = action_dim

    td = TensorDict(
        {
            "obs": torch.randn(batch_size, obs_dim),
            "obs_next": torch.randn(batch_size, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size,)),
        },
        batch_size=batch_size,
    )

    output_td = component.forward(td)

    # Compute a dummy loss using all outputs
    loss = (
        output_td["latent_mean"].sum()
        + output_td["obs_next_pred"].sum()
        + output_td["future_pred"].sum()  # Include auxiliary prediction
    )

    # Backward should work
    loss.backward()

    # Check that at least some gradients exist (not all params may be used)
    has_grads = any(param.grad is not None for param in component.parameters() if param.requires_grad)
    assert has_grads, "No gradients computed for any parameters"


def test_device_property(component):
    """Test device property."""
    device = component.device
    assert isinstance(device, torch.device)


def test_reset_memory(component):
    """Test reset_memory method."""
    # Should not raise an error
    component.reset_memory()


def test_get_agent_experience_spec(component):
    """Test get_agent_experience_spec method."""
    spec = component.get_agent_experience_spec()
    assert spec is not None


def test_no_auxiliary(config):
    """Test component with auxiliary task disabled."""
    config.use_auxiliary = False
    component = LatentDynamicsModelComponent(config=config, env=None)

    batch_size = 4
    latent_dim = component._latent_dim

    z = torch.randn(batch_size, latent_dim)
    future_pred = component.predict_future(z)

    assert future_pred is None


def test_unsupported_future_type(config):
    """Test that unsupported future_type raises an error during config validation."""
    from pydantic import ValidationError

    # Test 1: Setting an invalid future_type raises a Pydantic ValidationError
    with pytest.raises(ValidationError, match="future_type='observations' is not supported"):
        config.future_type = "observations"

    # Test 2: Creating a config with invalid future_type also raises an error
    with pytest.raises(ValidationError, match="future_type='observations' is not supported"):
        LatentDynamicsConfig(
            name="test_invalid",
            future_type="observations",
        )


def test_training_env_ids_size_mismatch(component):
    """Test that mismatched training_env_ids size raises a clear error."""
    batch_size = 4
    obs_dim = 64
    action_dim = 8

    component._action_dim = action_dim

    # Create TensorDict first without training_env_ids
    td = TensorDict(
        {
            "obs": torch.randn(batch_size, obs_dim),
            "actions": torch.randint(0, action_dim, (batch_size,)),
        },
        batch_size=batch_size,
    )

    # Manually set mismatched training_env_ids (bypassing TensorDict validation)
    # This simulates a bug where training_env_ids doesn't match batch size
    td._tensordict["training_env_ids"] = torch.tensor([[0], [1]])

    with pytest.raises(ValueError, match="training_env_ids has 2 elements but batch_size is 4"):
        component.forward(td)
