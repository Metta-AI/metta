"""Tests for Muesli algorithm losses (Model loss and CMPO)."""

import pytest
import torch
from tensordict import TensorDict

from metta.rl.loss.cmpo import CMPO, CMPOConfig
from metta.rl.loss.muesli import MuesliModel, MuesliModelConfig
from metta.rl.training import ComponentContext


class MockPolicy:
    """Mock policy for testing."""

    def __init__(self, num_actions: int = 4):
        self.num_actions = num_actions

    def parameters(self):
        """Return empty parameters for EMA."""
        return []

    def get_agent_experience_spec(self):
        """Return empty spec."""
        from torchrl.data import Composite

        return Composite()


class MockTrainerConfig:
    """Mock trainer config."""

    total_timesteps: int = 10000
    batch_size: int = 128


class MockEnv:
    """Mock environment."""

    @property
    def single_action_space(self):
        """Return mock action space."""
        import numpy as np

        class Space:
            dtype = np.int32

        return Space()


@pytest.fixture
def device():
    """Test device."""
    return torch.device("cpu")


@pytest.fixture
def mock_policy():
    """Create a mock policy."""
    return MockPolicy(num_actions=4)


@pytest.fixture
def mock_trainer_cfg():
    """Create mock trainer config."""
    return MockTrainerConfig()


@pytest.fixture
def mock_env():
    """Create mock environment."""
    return MockEnv()


@pytest.fixture
def context():
    """Create component context with mock components."""
    from unittest.mock import MagicMock

    mock_state = MagicMock()
    mock_state.epoch = 0
    mock_policy = MagicMock()
    mock_env = MagicMock()
    mock_experience = MagicMock()
    mock_optimizer = MagicMock()
    mock_config = MagicMock()
    mock_stopwatch = MagicMock()
    mock_distributed = MagicMock()

    ctx = ComponentContext(
        state=mock_state,
        policy=mock_policy,
        env=mock_env,
        experience=mock_experience,
        optimizer=mock_optimizer,
        config=mock_config,
        stopwatch=mock_stopwatch,
        distributed=mock_distributed,
    )
    # Don't set training_env_id - it's not needed for these tests
    return ctx


class TestMuesliModelLoss:
    """Tests for Muesli model loss."""

    def test_init(self, mock_policy, mock_trainer_cfg, mock_env, device):
        """Test initialization."""
        config = MuesliModelConfig(enabled=True, policy_horizon=5)
        loss = MuesliModel(
            mock_policy,
            mock_trainer_cfg,
            mock_env,
            device,
            instance_name="muesli_model",
            cfg=config,
        )

        assert loss.cfg.policy_horizon == 5
        assert loss.cfg.policy_pred_coef == 1.0
        # assert loss.target_policy is not None

    def test_init_without_target_network(self, mock_policy, mock_trainer_cfg, mock_env, device):
        """Test initialization without target network."""
        config = MuesliModelConfig(enabled=True)
        _ = MuesliModel(
            mock_policy,
            mock_trainer_cfg,
            mock_env,
            device,
            instance_name="muesli_model",
            cfg=config,
        )

        # assert loss.target_policy is None

    def test_run_train(self, mock_policy, mock_trainer_cfg, mock_env, device, context):
        """Test forward pass."""
        config = MuesliModelConfig(
            enabled=True,
            policy_horizon=3,
        )
        loss = MuesliModel(
            mock_policy,
            mock_trainer_cfg,
            mock_env,
            device,
            instance_name="muesli_model",
            cfg=config,
        )

        # Create mock data
        B, T = 4, 10
        A = 4  # num actions

        minibatch = TensorDict(
            {
                "obs": torch.randn(B, T, 84, 84, 3),
                "actions": torch.randint(0, A, (B, T)),
                "rewards": torch.randn(B, T),
                "returns": torch.randn(B, T),
                "advantages": torch.randn(B, T),
                "values": torch.randn(B, T),
            },
            batch_size=(B, T),
        )

        policy_td = TensorDict(
            {
                "hidden_state": torch.randn(B, T, 64),
                "value_pred": torch.randn(B, T, 1),
                "logits": torch.randn(B, T, A),
                "reward_pred": torch.randn(B, T, 1),
            },
            batch_size=(B, T),
        )

        shared_loss_data = TensorDict(
            {
                "sampled_mb": minibatch,
                "policy_td": policy_td,
                "muesli_unrolled_logits": torch.randn(3, B, T - 3, A),  # Mock unrolled logits
            },
            batch_size=(),
        )

        # Run training step
        loss_value, updated_data, stop_flag = loss.run_train(shared_loss_data, context, mb_idx=0)

        # Verify output
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.ndim == 0  # scalar
        assert not stop_flag

        # Verify loss tracking
        # assert "muesli_model_loss" in loss.loss_tracker
        # assert "muesli_value_loss" in loss.loss_tracker
        # assert "muesli_reward_loss" in loss.loss_tracker
        assert "muesli_policy_loss" in loss.loss_tracker

    def test_update_target_network(self, mock_policy, mock_trainer_cfg, mock_env, device):
        """Test target network update."""
        # MuesliModel no longer manages target network directly
        pass
        """
        config = MuesliModelConfig(enabled=True)

        # Create a policy with actual parameters for testing
        class PolicyWithParams:
            def __init__(self):
                self.param = torch.nn.Parameter(torch.ones(10))

            def parameters(self):
                return [self.param]

            def get_agent_experience_spec(self):
                from torchrl.data import Composite

                return Composite()

        policy = PolicyWithParams()
        loss = MuesliModel(
            policy,
            mock_trainer_cfg,
            mock_env,
            device,
            instance_name="muesli_model",
            cfg=config,
        )

        # Store initial target parameter
        initial_target = loss.target_policy.param.clone()

        # Update online parameter
        policy.param.data.fill_(2.0)

        # Update target
        loss.update_target_network()

        # Verify EMA update
        expected = 0.99 * initial_target + 0.01 * policy.param
        torch.testing.assert_close(loss.target_policy.param, expected)
        """


class TestCMPO:
    """Tests for CMPO loss."""

    def test_init(self, mock_policy, mock_trainer_cfg, mock_env, device):
        """Test initialization."""
        config = CMPOConfig(enabled=True, kl_coef=0.1)
        loss = CMPO(
            mock_policy,
            mock_trainer_cfg,
            mock_env,
            device,
            instance_name="cmpo",
            cfg=config,
        )

        assert loss.cfg.kl_coef == 0.1
        assert loss.target_policy is not None
        assert loss.advantage_variance_ema == 1.0

    def test_compute_cmpo_policy(self, mock_policy, mock_trainer_cfg, mock_env, device):
        """Test CMPO policy computation."""
        config = CMPOConfig(
            enabled=True,
            advantage_clip_min=-1.0,
            advantage_clip_max=1.0,
        )
        loss = CMPO(
            mock_policy,
            mock_trainer_cfg,
            mock_env,
            device,
            instance_name="cmpo",
            cfg=config,
        )

        B, T, A = 4, 10, 4
        logits = torch.randn(B, T, A)
        advantages = torch.randn(B, T)

        # Compute CMPO policy
        pi_cmpo = loss.compute_cmpo_policy(logits, advantages)

        # Verify output shape
        assert pi_cmpo.shape == (B, T, A)

        # Verify it's a valid probability distribution
        torch.testing.assert_close(pi_cmpo.sum(dim=-1), torch.ones(B, T))
        assert (pi_cmpo >= 0).all()
        assert (pi_cmpo <= 1).all()

    def test_compute_cmpo_policy_with_action_advantages(self, mock_policy, mock_trainer_cfg, mock_env, device):
        """Test CMPO policy computation with per-action advantages."""
        config = CMPOConfig(enabled=True)
        loss = CMPO(
            mock_policy,
            mock_trainer_cfg,
            mock_env,
            device,
            instance_name="cmpo",
            cfg=config,
        )

        B, T, A = 4, 10, 4
        logits = torch.randn(B, T, A)
        advantages = torch.randn(B, T, A)  # Per-action advantages

        # Compute CMPO policy
        pi_cmpo = loss.compute_cmpo_policy(logits, advantages)

        # Verify output
        assert pi_cmpo.shape == (B, T, A)
        torch.testing.assert_close(pi_cmpo.sum(dim=-1), torch.ones(B, T), atol=1e-6, rtol=1e-5)

    def test_run_train(self, mock_policy, mock_trainer_cfg, mock_env, device, context):
        """Test CMPO training step."""
        config = CMPOConfig(enabled=True, kl_coef=0.1, use_target_network=False)
        loss = CMPO(
            mock_policy,
            mock_trainer_cfg,
            mock_env,
            device,
            instance_name="cmpo",
            cfg=config,
        )

        # Create mock data
        B, T = 4, 10
        A = 4

        minibatch = TensorDict(
            {
                "advantages": torch.randn(B, T),
            },
            batch_size=(B, T),
        )

        policy_td = TensorDict(
            {
                "logits": torch.randn(B, T, A),
            },
            batch_size=(B, T),
        )

        shared_loss_data = TensorDict(
            {
                "sampled_mb": minibatch,
                "policy_td": policy_td,
                "muesli_unrolled_logits": torch.randn(3, B, T - 3, A),  # Mock unrolled logits
            },
            batch_size=(),
        )

        # Run training step
        loss_value, updated_data, stop_flag = loss.run_train(shared_loss_data, context, mb_idx=0)

        # Verify output
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.ndim == 0
        assert not stop_flag

        # Verify CMPO policies are stored in shared data
        assert "cmpo_policies" in updated_data
        assert updated_data["cmpo_policies"].shape == (B, T, A)

        # Verify loss tracking
        assert "cmpo_kl_loss" in loss.loss_tracker
        assert "cmpo_loss" in loss.loss_tracker
        assert "cmpo_adv_mean" in loss.loss_tracker
        assert "cmpo_adv_std" in loss.loss_tracker

    def test_advantage_variance_ema_update(self, mock_policy, mock_trainer_cfg, mock_env, device, context):
        """Test that advantage variance EMA is updated."""
        config = CMPOConfig(enabled=True, use_target_network=False)
        loss = CMPO(
            mock_policy,
            mock_trainer_cfg,
            mock_env,
            device,
            instance_name="cmpo",
            cfg=config,
        )

        initial_variance = loss.advantage_variance_ema.clone()

        # Create data with high variance advantages
        B, T, A = 4, 10, 4
        advantages = torch.randn(B, T) * 10.0  # High variance
        logits = torch.randn(B, T, A)

        # Compute CMPO policy (this updates the variance EMA)
        loss.compute_cmpo_policy(logits, advantages)

        # Verify EMA was updated
        assert not torch.allclose(loss.advantage_variance_ema, initial_variance)
        assert loss.advantage_variance_ema > 0


class TestMuesliIntegration:
    """Integration tests for Muesli components."""

    def test_muesli_with_cmpo_target_policies(self, mock_policy, mock_trainer_cfg, mock_env, device, context):
        """Test that Muesli model loss can use CMPO target policies."""
        # Create both losses
        cmpo = CMPO(
            mock_policy,
            mock_trainer_cfg,
            mock_env,
            device,
            instance_name="cmpo",
            cfg=CMPOConfig(enabled=True, use_target_network=False),
        )

        muesli = MuesliModel(
            mock_policy,
            mock_trainer_cfg,
            mock_env,
            device,
            instance_name="muesli_model",
            cfg=MuesliModelConfig(enabled=True, policy_horizon=3),
        )

        # Create mock data
        B, T, A = 4, 10, 4

        minibatch = TensorDict(
            {
                "obs": torch.randn(B, T, 84, 84, 3),
                "actions": torch.randint(0, A, (B, T)),
                "rewards": torch.randn(B, T),
                "returns": torch.randn(B, T),
                "advantages": torch.randn(B, T),
                "values": torch.randn(B, T),
            },
            batch_size=(B, T),
        )

        policy_td = TensorDict(
            {
                "hidden_state": torch.randn(B, T, 64),
                "value_pred": torch.randn(B, T, 1),
                "logits": torch.randn(B, T, A),
                "reward_pred": torch.randn(B, T, 1),
            },
            batch_size=(B, T),
        )

        shared_loss_data = TensorDict(
            {
                "sampled_mb": minibatch,
                "policy_td": policy_td,
                "muesli_unrolled_logits": torch.randn(3, B, T - 3, A),  # Mock unrolled logits
            },
            batch_size=(),
        )

        # Run CMPO first to compute target policies
        _, shared_loss_data, _ = cmpo.run_train(shared_loss_data, context, mb_idx=0)

        # Verify CMPO policies are in shared data
        assert "cmpo_policies" in shared_loss_data

        # Run Muesli model loss with CMPO policies
        loss_value, _, _ = muesli.run_train(shared_loss_data, context, mb_idx=0)

        # Verify loss computation succeeded
        assert isinstance(loss_value, torch.Tensor)
        assert not torch.isnan(loss_value)
        assert not torch.isinf(loss_value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
