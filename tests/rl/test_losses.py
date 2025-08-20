"""
Unit tests for the Losses class in metta.rl.losses.

This module provides tests for the Losses class, which is responsible for
tracking and computing various loss metrics during training.
"""

import pytest

from metta.rl.losses import Losses


class TestLosses:
    """Test suite for the Losses class."""

    def test_initialization(self):
        """Test that a new Losses instance is properly initialized with zeros."""
        losses = Losses()

        # Check that all loss values are initialized to zero
        assert losses.policy_loss_sum == 0.0
        assert losses.value_loss_sum == 0.0
        assert losses.entropy_sum == 0.0
        assert losses.approx_kl_sum == 0.0
        assert losses.clipfrac_sum == 0.0
        assert losses.l2_reg_loss_sum == 0.0
        assert losses.l2_init_loss_sum == 0.0
        assert losses.ks_action_loss_sum == 0.0
        assert losses.ks_value_loss_sum == 0.0
        assert losses.importance_sum == 0.0
        assert losses.explained_variance == 0.0
        assert losses.minibatches_processed == 0

    def test_zero_method(self):
        """Test that the zero method resets all loss values."""
        losses = Losses()

        # Set some non-zero values
        losses.policy_loss_sum = 1.0
        losses.value_loss_sum = 2.0
        losses.entropy_sum = 3.0
        losses.minibatches_processed = 5

        # Reset values
        losses.zero()

        # Check that all values are reset to zero
        assert losses.policy_loss_sum == 0.0
        assert losses.value_loss_sum == 0.0
        assert losses.entropy_sum == 0.0
        assert losses.minibatches_processed == 0

    def test_stats_with_no_minibatches(self):
        """Test stats method when no minibatches have been processed."""
        losses = Losses()

        # Set some non-zero values
        losses.policy_loss_sum = 10.0
        losses.value_loss_sum = 20.0
        losses.entropy_sum = 5.0

        # Get stats
        stats = losses.stats()

        # Since minibatches_processed is 0, it should use n=1 for division
        assert stats["policy_loss"] == 10.0
        assert stats["value_loss"] == 20.0
        assert stats["entropy"] == 5.0

    def test_stats_with_minibatches(self):
        """Test stats method when multiple minibatches have been processed."""
        losses = Losses()

        # Set some non-zero values
        losses.policy_loss_sum = 10.0
        losses.value_loss_sum = 20.0
        losses.entropy_sum = 5.0
        losses.approx_kl_sum = 2.0
        losses.clipfrac_sum = 0.5
        losses.l2_reg_loss_sum = 1.0
        losses.l2_init_loss_sum = 0.3
        losses.ks_action_loss_sum = 0.7
        losses.ks_value_loss_sum = 0.9
        losses.importance_sum = 1.5
        losses.explained_variance = 0.8
        losses.minibatches_processed = 5

        # Get stats
        stats = losses.stats()

        # Check that values are properly averaged
        assert stats["policy_loss"] == pytest.approx(2.0)  # 10.0 / 5
        assert stats["value_loss"] == pytest.approx(4.0)  # 20.0 / 5
        assert stats["entropy"] == pytest.approx(1.0)  # 5.0 / 5
        assert stats["approx_kl"] == pytest.approx(0.4)  # 2.0 / 5
        assert stats["clipfrac"] == pytest.approx(0.1)  # 0.5 / 5
        assert stats["l2_reg_loss"] == pytest.approx(0.2)  # 1.0 / 5
        assert stats["l2_init_loss"] == pytest.approx(0.06)  # 0.3 / 5
        assert stats["ks_action_loss"] == pytest.approx(0.14)  # 0.7 / 5
        assert stats["ks_value_loss"] == pytest.approx(0.18)  # 0.9 / 5
        assert stats["importance"] == pytest.approx(0.3)  # 1.5 / 5
        assert stats["explained_variance"] == pytest.approx(0.8)  # Not averaged

    def test_stats_return_type(self):
        """Test that stats method returns a dictionary with the correct keys."""
        losses = Losses()
        stats = losses.stats()

        # Check that the return value is a dictionary
        assert isinstance(stats, dict)

        # Check that all expected keys are present
        expected_keys = [
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clipfrac",
            "l2_reg_loss",
            "l2_init_loss",
            "ks_action_loss",
            "ks_value_loss",
            "importance",
            "explained_variance",
        ]
        for key in expected_keys:
            assert key in stats


def test_masking_vs_slicing_equivalence_for_ppo_loss():
    import torch
    from tensordict import TensorDict

    from metta.rl.ppo import compute_ppo_losses
    from metta.rl.trainer_config import (
        CheckpointConfig,
        PPOConfig,
        SimulationConfig,
        TorchProfilerConfig,
        TrainerConfig,
    )

    # Simulate a batch of 6 agents: 4 students, 2 NPCs
    batch_size = 6
    num_students = 4
    is_student_agent = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.float32)
    student_indices = is_student_agent.nonzero(as_tuple=True)[0]

    # Create dummy minibatch
    minibatch = TensorDict(
        {
            "returns": torch.tensor([1.5, 2.5, 2.5, 3.5, 100.0, 200.0]),
            "values": torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0], requires_grad=True),
            "act_log_prob": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        },
        batch_size=[batch_size],
    )
    new_logprob = torch.tensor([0.15, 0.25, 0.35, 0.45, 0.55, 0.65], requires_grad=True)
    entropy = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], requires_grad=True)
    newvalue = minibatch["values"]  # Use same tensor for simplicity
    importance_sampling_ratio = torch.tensor([1.0, 1.1, 0.9, 1.05, 1.0, 1.0], requires_grad=True)
    adv = torch.tensor([0.2, 0.3, 0.1, 0.4, 0.0, 0.0], requires_grad=True)

    # Trainer config (provide required fields)
    trainer_cfg = TrainerConfig(
        ppo=PPOConfig(),
        num_workers=1,
        profiler=TorchProfilerConfig(profile_dir="/tmp"),
        checkpoint=CheckpointConfig(checkpoint_dir="/tmp"),
        simulation=SimulationConfig(replay_dir="/tmp"),
    )

    # --- Masking approach ---
    masked_new_logprob = new_logprob * is_student_agent
    masked_entropy = entropy * is_student_agent
    masked_adv = adv * is_student_agent
    masked_importance_ratio = importance_sampling_ratio * is_student_agent + (1 - is_student_agent)
    # Add mask to minibatch for completeness
    minibatch_masked = minibatch.clone()
    minibatch_masked["is_student_agent"] = is_student_agent

    # Compute losses as usual
    pg_loss_mask, v_loss_mask, entropy_loss_mask, approx_kl_mask, clipfrac_mask = compute_ppo_losses(
        minibatch_masked,
        masked_new_logprob,
        masked_entropy,
        newvalue,
        masked_importance_ratio,
        masked_adv,
        trainer_cfg,
    )
    # Correct the mean: sum over students / num_students
    pg_loss_mask = (pg_loss_mask * is_student_agent).sum() / num_students
    v_loss_mask = (v_loss_mask * is_student_agent).sum() / num_students
    entropy_loss_mask = (entropy_loss_mask * is_student_agent).sum() / num_students
    total_loss_mask = pg_loss_mask + v_loss_mask + entropy_loss_mask
    total_loss_mask.backward()
    grads_mask = newvalue.grad.clone() if newvalue.grad is not None else None
    # Reset grads
    if newvalue.grad is not None:
        newvalue.grad.zero_()
    if new_logprob.grad is not None:
        new_logprob.grad.zero_()
    if entropy.grad is not None:
        entropy.grad.zero_()
    if adv.grad is not None:
        adv.grad.zero_()
    if importance_sampling_ratio.grad is not None:
        importance_sampling_ratio.grad.zero_()

    # --- Slicing approach ---
    sliced_new_logprob = new_logprob[student_indices]
    sliced_entropy = entropy[student_indices]
    sliced_adv = adv[student_indices]
    sliced_importance_ratio = importance_sampling_ratio[student_indices]
    sliced_newvalue = newvalue[student_indices]
    sliced_minibatch = TensorDict(
        {
            "returns": minibatch["returns"][student_indices],
            "values": minibatch["values"][student_indices],
            "act_log_prob": minibatch["act_log_prob"][student_indices],
        },
        batch_size=[num_students],
    )

    pg_loss_slice, v_loss_slice, entropy_loss_slice, approx_kl_slice, clipfrac_slice = compute_ppo_losses(
        sliced_minibatch,
        sliced_new_logprob,
        sliced_entropy,
        sliced_newvalue,
        sliced_importance_ratio,
        sliced_adv,
        trainer_cfg,
    )
    total_loss_slice = pg_loss_slice + v_loss_slice + entropy_loss_slice
    total_loss_slice.backward()
    grads_slice = newvalue.grad.clone() if newvalue.grad is not None else None

    # --- Compare results ---
    assert torch.allclose(pg_loss_mask, pg_loss_slice, atol=1e-6)
    assert torch.allclose(v_loss_mask, v_loss_slice, atol=1e-6)
    assert torch.allclose(entropy_loss_mask, entropy_loss_slice, atol=1e-6)
    assert torch.allclose(total_loss_mask, total_loss_slice, atol=1e-6)
    # Gradients for NPC entries should be zero
    assert grads_mask is not None and grads_slice is not None
    assert torch.allclose(grads_mask[:num_students], grads_slice[:num_students], atol=1e-6)
    assert torch.all(grads_mask[num_students:] == 0)
    assert torch.all(grads_slice[num_students:] == 0)
