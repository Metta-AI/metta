"""Tests for diversity injection component and loss."""

import torch
from tensordict import TensorDict

from metta.agent.components.diversity_injection import DiversityInjection, DiversityInjectionConfig


class TestDiversityInjection:
    """Tests for DiversityInjection component."""

    def test_basic_forward(self):
        """Test basic forward pass with agent IDs."""
        config = DiversityInjectionConfig(
            in_key="hidden",
            out_key="hidden_div",
            num_agents=10,
            projection_rank=8,
        )
        layer = DiversityInjection(config)

        batch_size = 4
        hidden_dim = 32
        hidden = torch.randn(batch_size, hidden_dim)
        agent_ids = torch.tensor([[0], [1], [2], [3]])

        td = TensorDict(
            {"hidden": hidden, "training_env_ids": agent_ids},
            batch_size=[batch_size],
        )

        result = layer(td)

        assert "hidden_div" in result.keys()
        assert result["hidden_div"].shape == hidden.shape
        # Alpha is accessible via the layer property
        assert layer.alpha.item() > 0

    def test_different_agents_different_perturbations(self):
        """Test that different agents get different perturbations."""
        config = DiversityInjectionConfig(
            in_key="hidden",
            out_key="hidden_div",
            num_agents=10,
            projection_rank=8,
            log_alpha_init=0.0,  # α = 1.0 for clearer differences
            use_layer_norm=False,  # Disable to see raw differences
        )
        layer = DiversityInjection(config)

        # Same input, different agent IDs
        hidden = torch.ones(2, 32)
        agent_ids = torch.tensor([[0], [1]])

        td = TensorDict(
            {"hidden": hidden, "training_env_ids": agent_ids},
            batch_size=[2],
        )

        result = layer(td)

        # Different agents should produce different outputs
        assert not torch.allclose(result["hidden_div"][0], result["hidden_div"][1])

    def test_same_agent_same_perturbation(self):
        """Test that same agent ID produces consistent perturbations."""
        config = DiversityInjectionConfig(
            in_key="hidden",
            out_key="hidden_div",
            num_agents=10,
            projection_rank=8,
            use_layer_norm=False,
        )
        layer = DiversityInjection(config)

        hidden = torch.ones(2, 32)
        agent_ids = torch.tensor([[5], [5]])  # Same agent

        td = TensorDict(
            {"hidden": hidden, "training_env_ids": agent_ids},
            batch_size=[2],
        )

        result = layer(td)

        # Same agent, same input -> same output
        assert torch.allclose(result["hidden_div"][0], result["hidden_div"][1])

    def test_alpha_clamping(self):
        """Test that alpha is properly clamped."""
        config = DiversityInjectionConfig(
            in_key="hidden",
            out_key="hidden_div",
            log_alpha_init=10.0,  # Very large
            alpha_max=2.0,
        )
        layer = DiversityInjection(config)

        assert layer.alpha.item() <= 2.0

    def test_diversity_loss(self):
        """Test diversity loss computation."""
        config = DiversityInjectionConfig(
            in_key="hidden",
            out_key="hidden_div",
            log_alpha_init=-1.0,
        )
        layer = DiversityInjection(config)

        loss = layer.get_diversity_loss()

        # -log_alpha where log_alpha = -1.0 -> loss = 1.0
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_gradient_flow(self):
        """Test that gradients flow through log_alpha."""
        config = DiversityInjectionConfig(
            in_key="hidden",
            out_key="hidden_div",
        )
        layer = DiversityInjection(config)

        hidden = torch.randn(4, 32)
        agent_ids = torch.tensor([[0], [1], [2], [3]])

        td = TensorDict(
            {"hidden": hidden, "training_env_ids": agent_ids},
            batch_size=[4],
        )

        layer(td)
        loss = layer.get_diversity_loss()
        loss.backward()

        assert layer.log_alpha.grad is not None

    def test_3d_input(self):
        """Test handling of (batch, time, hidden) inputs."""
        config = DiversityInjectionConfig(
            in_key="hidden",
            out_key="hidden_div",
            num_agents=10,
        )
        layer = DiversityInjection(config)

        batch_size = 4
        time_steps = 8
        hidden_dim = 32
        hidden = torch.randn(batch_size, time_steps, hidden_dim)
        agent_ids = torch.tensor([[0], [1], [2], [3]])

        td = TensorDict(
            {"hidden": hidden, "training_env_ids": agent_ids},
            batch_size=[batch_size],
        )

        result = layer(td)

        assert result["hidden_div"].shape == (batch_size, time_steps, hidden_dim)

    def test_no_agent_ids_defaults_to_zero(self):
        """Test that missing agent IDs default to agent 0."""
        config = DiversityInjectionConfig(
            in_key="hidden",
            out_key="hidden_div",
        )
        layer = DiversityInjection(config)

        hidden = torch.randn(4, 32)
        td = TensorDict({"hidden": hidden}, batch_size=[4])

        # Should not raise
        result = layer(td)
        assert "hidden_div" in result.keys()

    def test_layer_norm_applied(self):
        """Test that LayerNorm is applied when enabled."""
        config = DiversityInjectionConfig(
            in_key="hidden",
            out_key="hidden_div",
            use_layer_norm=True,
            log_alpha_init=2.0,  # Larger alpha to see effect
        )
        layer = DiversityInjection(config)

        hidden = torch.randn(4, 32)
        agent_ids = torch.tensor([[0], [1], [2], [3]])

        td = TensorDict(
            {"hidden": hidden, "training_env_ids": agent_ids},
            batch_size=[4],
        )

        layer(td)

        # LayerNorm should be initialized after first forward
        assert layer.layer_norm is not None
