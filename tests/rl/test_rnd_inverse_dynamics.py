"""Tests for RND and Inverse Dynamics losses."""

import torch
import torch.nn as nn

from metta.rl.loss.inverse_dynamics import InverseDynamicsConfig, InverseDynamicsPredictor
from metta.rl.loss.rnd import RNDConfig, RNDPredictorNetwork, RNDTargetNetwork


class TestRND:
    """Tests for Random Network Distillation."""

    def test_target_network_frozen(self):
        """Test that target network parameters are frozen."""
        target = RNDTargetNetwork(input_dim=64, hidden_dim=128, output_dim=32)

        for param in target.parameters():
            assert not param.requires_grad

    def test_predictor_network_trainable(self):
        """Test that predictor network parameters are trainable."""
        predictor = RNDPredictorNetwork(input_dim=64, hidden_dim=128, output_dim=32)

        for param in predictor.parameters():
            assert param.requires_grad

    def test_networks_output_shape(self):
        """Test output shapes of target and predictor networks."""
        input_dim, hidden_dim, output_dim = 64, 128, 32
        batch_size = 16

        target = RNDTargetNetwork(input_dim, hidden_dim, output_dim)
        predictor = RNDPredictorNetwork(input_dim, hidden_dim, output_dim)

        x = torch.randn(batch_size, input_dim)

        target_out = target(x)
        predictor_out = predictor(x)

        assert target_out.shape == (batch_size, output_dim)
        assert predictor_out.shape == (batch_size, output_dim)

    def test_target_deterministic(self):
        """Test that target network gives same output for same input."""
        target = RNDTargetNetwork(input_dim=64, hidden_dim=128, output_dim=32)

        x = torch.randn(16, 64)

        out1 = target(x)
        out2 = target(x)

        assert torch.allclose(out1, out2)

    def test_prediction_error_computable(self):
        """Test that prediction error can be computed."""
        target = RNDTargetNetwork(input_dim=64, hidden_dim=128, output_dim=32)
        predictor = RNDPredictorNetwork(input_dim=64, hidden_dim=128, output_dim=32)

        x = torch.randn(16, 64)

        with torch.no_grad():
            target_out = target(x)

        predictor_out = predictor(x)

        # MSE as intrinsic reward
        intrinsic = ((predictor_out - target_out) ** 2).mean(dim=-1)

        assert intrinsic.shape == (16,)
        assert intrinsic.min() >= 0

    def test_config_defaults(self):
        """Test RND config default values."""
        cfg = RNDConfig()

        # Note: enabled=True is base LossConfig default, but LossesConfig overrides to False
        assert cfg.enabled is True
        assert cfg.rnd_coef == 0.1
        assert cfg.intrinsic_reward_coef == 0.01
        assert cfg.hidden_dim == 256
        assert cfg.output_dim == 64


class TestInverseDynamics:
    """Tests for Inverse Dynamics loss."""

    def test_predictor_output_shape(self):
        """Test predictor output shape."""
        input_dim, hidden_dim, num_actions = 64, 128, 10
        batch_size = 16

        predictor = InverseDynamicsPredictor(input_dim, hidden_dim, num_actions)

        s_t = torch.randn(batch_size, input_dim)
        s_t1 = torch.randn(batch_size, input_dim)

        logits = predictor(s_t, s_t1)

        assert logits.shape == (batch_size, num_actions)

    def test_predictor_trainable(self):
        """Test that predictor is trainable."""
        predictor = InverseDynamicsPredictor(64, 128, 10)

        for param in predictor.parameters():
            assert param.requires_grad

    def test_cross_entropy_loss(self):
        """Test that cross-entropy loss can be computed."""
        predictor = InverseDynamicsPredictor(64, 128, 10)

        s_t = torch.randn(16, 64)
        s_t1 = torch.randn(16, 64)
        actions = torch.randint(0, 10, (16,))

        logits = predictor(s_t, s_t1)
        loss = nn.functional.cross_entropy(logits, actions)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient_flow(self):
        """Test gradients flow through predictor."""
        predictor = InverseDynamicsPredictor(64, 128, 10)

        s_t = torch.randn(16, 64)
        s_t1 = torch.randn(16, 64)
        actions = torch.randint(0, 10, (16,))

        logits = predictor(s_t, s_t1)
        loss = nn.functional.cross_entropy(logits, actions)
        loss.backward()

        for param in predictor.parameters():
            assert param.grad is not None

    def test_config_defaults(self):
        """Test Inverse Dynamics config default values."""
        cfg = InverseDynamicsConfig()

        # Note: enabled=True is base LossConfig default, but LossesConfig overrides to False
        assert cfg.enabled is True
        assert cfg.inv_dyn_coef == 0.1
        assert cfg.hidden_dim == 256
        assert cfg.use_encoder_features is True

    def test_accuracy_computation(self):
        """Test accuracy can be computed."""
        predictor = InverseDynamicsPredictor(64, 128, 10)

        s_t = torch.randn(16, 64)
        s_t1 = torch.randn(16, 64)
        actions = torch.randint(0, 10, (16,))

        logits = predictor(s_t, s_t1)
        predicted = logits.argmax(dim=-1)
        accuracy = (predicted == actions).float().mean()

        assert 0 <= accuracy <= 1
