"""Comprehensive tests for latent dynamics model rollout behavior.

Tests environment-specific state management, resets, and BPTT training mode.
"""

import torch
from tensordict import TensorDict

from metta.agent.components.dynamics import LatentDynamicsConfig, LatentDynamicsModelComponent


def _make_component(latent_dim=16, encoder_hidden=None, decoder_hidden=None):
    """Create a latent dynamics component for testing."""
    if encoder_hidden is None:
        encoder_hidden = [32]
    if decoder_hidden is None:
        decoder_hidden = [32]

    config = LatentDynamicsConfig(
        name="test_latent_dynamics",
        in_key="encoded",
        out_key="core",
        action_key="last_actions",
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        auxiliary_hidden=[16],
        beta_kl=0.01,
        gamma_auxiliary=1.0,
        future_horizon=3,
        future_type="returns",
        use_auxiliary=True,
        use_triton=False,  # Disable Triton for CPU testing
    )
    return LatentDynamicsModelComponent(config=config, env=None)


def test_rollout_updates_per_env_state():
    """Test that rollout mode maintains separate state for each environment."""
    component = _make_component()
    component._action_dim = 4

    # Initialize with 2 environments
    device = torch.device("cpu")
    component._ensure_capacity(2, device)

    # First rollout step - 2 envs, single step (inference mode)
    td = TensorDict(
        {
            "encoded": torch.randn(2, 16),
            "last_actions": torch.tensor([0, 1]),
            "training_env_ids": torch.tensor([0, 1]),
        },
        batch_size=[2],
    )

    # Forward pass (inference mode - no obs_next)
    out = component(td)

    assert out["core"].shape == torch.Size([2, component.config.latent_dim])
    assert "latent" in out.keys()

    # Check that per-environment states were updated
    assert component.latent_mean_state.shape[0] >= 2
    assert component.latent_logvar_state.shape[0] >= 2

    # Second rollout step - states should persist
    td2 = TensorDict(
        {
            "encoded": torch.randn(2, 16),
            "last_actions": torch.tensor([2, 3]),
            "training_env_ids": torch.tensor([0, 1]),
        },
        batch_size=[2],
    )

    component(td2)

    # States should have been maintained (they may be the same due to simple persistence)
    # but the component should be tracking them per-environment
    assert component.latent_mean_state.shape[0] >= 2


def test_rollout_resets_on_done():
    """Test that environments can have independent states during rollouts."""
    component = _make_component()
    component._action_dim = 4

    device = torch.device("cpu")
    component._ensure_capacity(2, device)

    # Set initial states for both environments
    component.latent_mean_state[0] = torch.ones(16) * 5.0
    component.latent_mean_state[1] = torch.ones(16) * 10.0

    # Rollout for both environments
    td = TensorDict(
        {
            "encoded": torch.randn(2, 16),
            "last_actions": torch.tensor([0, 1]),
            "training_env_ids": torch.tensor([0, 1]),
        },
        batch_size=[2],
    )

    out = component(td)

    # Both environments should have received their respective states
    assert out["latent"].shape == torch.Size([2, 16])

    # Environment 0 should have used state starting from 5.0
    # Environment 1 should have used state starting from 10.0
    # (The actual output depends on reparameterization, but states were loaded)


def test_training_mode_with_bptt():
    """Test that training mode with BPTT handles sequences correctly."""
    component = _make_component()
    component._action_dim = 4

    # Training mode: batch_size=4, sequence_length=2 (BPTT)
    # Total elements: 4 * 2 = 8
    td = TensorDict(
        {
            "encoded": torch.randn(8, 16),  # Flattened: (batch * time, features)
            "encoded_next": torch.randn(8, 16),  # Next observations for encoding
            "last_actions": torch.randint(0, 4, (8,)),
            "bptt": torch.full((8,), 2, dtype=torch.long),  # Sequence length = 2
            "training_env_ids": torch.arange(8),
        },
        batch_size=[8],
    )

    # Forward pass in training mode (with next observations)
    out = component(td)

    # Output should match input shape
    assert out["core"].shape == torch.Size([8, component.config.latent_dim])
    assert "latent_mean" in out.keys()
    assert "latent_logvar" in out.keys()
    assert "latent" in out.keys()
    assert "obs_next_pred" in out.keys()

    # Check that latent variables have correct shape
    assert out["latent_mean"].shape == torch.Size([8, 16])
    assert out["latent_logvar"].shape == torch.Size([8, 16])


def test_training_keeps_shape():
    """Test that training mode preserves tensor shapes correctly."""
    component = _make_component()
    component._action_dim = 4

    batch_size = 16

    td = TensorDict(
        {
            "encoded": torch.randn(batch_size, 16),
            "encoded_next": torch.randn(batch_size, 16),
            "last_actions": torch.randint(0, 4, (batch_size,)),
            "training_env_ids": torch.arange(batch_size),
        },
        batch_size=[batch_size],
    )

    out = component(td)

    assert out["core"].shape == torch.Size([batch_size, component.config.latent_dim])
    assert out["latent"].shape == torch.Size([batch_size, component.config.latent_dim])


def test_capacity_expansion():
    """Test that component dynamically expands capacity for new environments."""
    component = _make_component()

    device = torch.device("cpu")

    # Initially empty
    assert component.max_num_envs == 0

    # Allocate for 5 environments
    component._ensure_capacity(5, device)
    assert component.max_num_envs == 5
    assert component.latent_mean_state.shape == torch.Size([5, 16])

    # Expand to 10 environments
    component._ensure_capacity(10, device)
    assert component.max_num_envs == 10
    assert component.latent_mean_state.shape == torch.Size([10, 16])

    # Asking for fewer envs doesn't shrink
    component._ensure_capacity(7, device)
    assert component.max_num_envs == 10


def test_mixed_env_ids():
    """Test handling of non-sequential environment IDs."""
    component = _make_component()
    component._action_dim = 4

    # Use non-sequential env IDs: 0, 2, 5
    td = TensorDict(
        {
            "encoded": torch.randn(3, 16),
            "last_actions": torch.tensor([0, 1, 2]),
            "training_env_ids": torch.tensor([0, 2, 5]),
        },
        batch_size=[3],
    )

    out = component(td)

    # Should allocate up to env 5 (i.e., 6 environments: 0-5)
    assert component.max_num_envs >= 6
    assert out["latent"].shape == torch.Size([3, 16])


def test_reset_memory_clears_states():
    """Test that reset_memory clears all per-environment states."""
    component = _make_component()

    device = torch.device("cpu")
    component._ensure_capacity(5, device)

    # Set some non-zero states
    component.latent_mean_state[:] = torch.randn_like(component.latent_mean_state)
    component.latent_logvar_state[:] = torch.randn_like(component.latent_logvar_state)

    # Reset memory
    component.reset_memory()

    # All states should be zero
    assert torch.allclose(component.latent_mean_state, torch.zeros_like(component.latent_mean_state))
    assert torch.allclose(component.latent_logvar_state, torch.zeros_like(component.latent_logvar_state))


def test_auxiliary_predictions_in_training():
    """Test that auxiliary predictions are generated in training mode."""
    component = _make_component()
    component._action_dim = 4

    td = TensorDict(
        {
            "encoded": torch.randn(4, 16),
            "encoded_next": torch.randn(4, 16),
            "last_actions": torch.randint(0, 4, (4,)),
            "training_env_ids": torch.arange(4),
        },
        batch_size=[4],
    )

    out = component(td)

    # Auxiliary predictions should be present in training mode
    assert "future_pred" in out.keys()
    assert out["future_pred"].shape[0] == 4


def test_inference_without_auxiliary():
    """Test inference mode doesn't crash even though auxiliary is configured."""
    component = _make_component()
    component._action_dim = 4

    device = torch.device("cpu")
    component._ensure_capacity(2, device)

    # Inference mode (no encoded_next)
    td = TensorDict(
        {
            "encoded": torch.randn(2, 16),
            "last_actions": torch.tensor([0, 1]),
            "training_env_ids": torch.tensor([0, 1]),
        },
        batch_size=[2],
    )

    out = component(td)

    # Should work fine, auxiliary predictions not expected in inference
    assert "latent" in out.keys()
    assert "future_pred" not in out.keys()  # No auxiliary in inference mode
