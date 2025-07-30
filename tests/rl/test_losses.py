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
