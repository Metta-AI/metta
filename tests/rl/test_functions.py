import pytest

from metta.rl.functions import calculate_l2_init_coef


class TestCalculateL2InitCoef:
    """Test L2-init coefficient annealing function."""

    def test_no_annealing_when_steps_zero(self):
        """Test that no annealing occurs when l2_init_anneal_steps is 0."""
        coef = calculate_l2_init_coef(
            agent_step=1000,
            l2_init_loss_coef=0.5,
            l2_init_anneal_steps=0,
            l2_init_anneal_ratio=0.2,
        )
        assert coef == 0.5  # Should return original coefficient

    def test_full_coefficient_before_ramp_down(self):
        """Test that full coefficient is returned before ramp-down phase."""
        coef = calculate_l2_init_coef(
            agent_step=500,  # Before ramp-down (starts at 800)
            l2_init_loss_coef=0.5,
            l2_init_anneal_steps=1000,
            l2_init_anneal_ratio=0.2,  # Last 200 steps for ramp-down
        )
        assert coef == 0.5

    def test_zero_coefficient_after_annealing_period(self):
        """Test that coefficient is zero after annealing period ends."""
        coef = calculate_l2_init_coef(
            agent_step=1500,  # After annealing period (1000 steps)
            l2_init_loss_coef=0.5,
            l2_init_anneal_steps=1000,
            l2_init_anneal_ratio=0.2,
        )
        assert coef == 0.0

    def test_linear_ramp_down_during_annealing(self):
        """Test linear ramp-down during annealing phase."""
        # Test at halfway through ramp-down
        coef = calculate_l2_init_coef(
            agent_step=900,  # Halfway through ramp-down (800-1000)
            l2_init_loss_coef=0.4,
            l2_init_anneal_steps=1000,
            l2_init_anneal_ratio=0.2,
        )
        # Should be 50% of original coefficient
        assert coef == pytest.approx(0.2, abs=1e-6)

        # Test at start of ramp-down
        coef = calculate_l2_init_coef(
            agent_step=800,  # Start of ramp-down
            l2_init_loss_coef=0.4,
            l2_init_anneal_steps=1000,
            l2_init_anneal_ratio=0.2,
        )
        assert coef == pytest.approx(0.4, abs=1e-6)

        # Test near end of ramp-down
        coef = calculate_l2_init_coef(
            agent_step=999,  # Near end of ramp-down
            l2_init_loss_coef=0.4,
            l2_init_anneal_steps=1000,
            l2_init_anneal_ratio=0.2,
        )
        # Should be very close to 0
        assert coef == pytest.approx(0.002, abs=1e-6)

    def test_no_ramp_down_when_ratio_zero(self):
        """Test that no ramp-down occurs when anneal_ratio is 0."""
        coef = calculate_l2_init_coef(
            agent_step=999,  # Near end of annealing period
            l2_init_loss_coef=0.5,
            l2_init_anneal_steps=1000,
            l2_init_anneal_ratio=0.0,  # No ramp-down
        )
        assert coef == 0.5  # Should maintain full coefficient

        # But should drop to 0 after annealing period
        coef = calculate_l2_init_coef(
            agent_step=1000,
            l2_init_loss_coef=0.5,
            l2_init_anneal_steps=1000,
            l2_init_anneal_ratio=0.0,
        )
        assert coef == 0.0
