import pytest

from metta.rl.functions import calculate_l2_init_coef, calculate_l2_init_coef_new


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


class TestCalculateL2InitCoefNew:
    """Test L2-init coefficient annealing function with new configuration system."""

    def test_no_annealing_when_config_none(self):
        """Test that no annealing occurs when annealing_config is None."""
        coef = calculate_l2_init_coef_new(
            agent_step=1000,
            l2_init_loss_coef=0.5,
            annealing_config=None,
        )
        assert coef == 0.5  # Should return original coefficient

    def test_no_annealing_when_l2_init_none(self):
        """Test that no annealing occurs when l2_init schedule is None."""

        # Mock annealing config with None l2_init
        class MockAnnealingConfig:
            l2_init = None

        coef = calculate_l2_init_coef_new(
            agent_step=1000,
            l2_init_loss_coef=0.5,
            annealing_config=MockAnnealingConfig(),
        )
        assert coef == 0.5  # Should return original coefficient

    def test_before_annealing_start(self):
        """Test coefficient before annealing starts."""

        # Mock annealing config
        class MockL2InitSchedule:
            start_coef = 0.8
            end_coef = 0.1
            start_step = 1000
            end_step = 5000

        class MockAnnealingConfig:
            l2_init = MockL2InitSchedule()

        coef = calculate_l2_init_coef_new(
            agent_step=500,  # Before start_step
            l2_init_loss_coef=0.5,  # This should be ignored
            annealing_config=MockAnnealingConfig(),
        )
        assert coef == 0.8  # Should return start_coef

    def test_after_annealing_end(self):
        """Test coefficient after annealing ends."""

        # Mock annealing config
        class MockL2InitSchedule:
            start_coef = 0.8
            end_coef = 0.1
            start_step = 1000
            end_step = 5000

        class MockAnnealingConfig:
            l2_init = MockL2InitSchedule()

        coef = calculate_l2_init_coef_new(
            agent_step=6000,  # After end_step
            l2_init_loss_coef=0.5,  # This should be ignored
            annealing_config=MockAnnealingConfig(),
        )
        assert coef == 0.1  # Should return end_coef

    def test_linear_interpolation_during_annealing(self):
        """Test linear interpolation during annealing phase."""

        # Mock annealing config
        class MockL2InitSchedule:
            start_coef = 0.8
            end_coef = 0.2
            start_step = 1000
            end_step = 5000

        class MockAnnealingConfig:
            l2_init = MockL2InitSchedule()

        # Test at halfway point
        coef = calculate_l2_init_coef_new(
            agent_step=3000,  # Halfway between 1000 and 5000
            l2_init_loss_coef=0.5,  # This should be ignored
            annealing_config=MockAnnealingConfig(),
        )
        expected = 0.8 + 0.5 * (0.2 - 0.8)  # 0.8 + 0.5 * (-0.6) = 0.5
        assert coef == pytest.approx(expected, abs=1e-6)

        # Test at 75% through annealing
        coef = calculate_l2_init_coef_new(
            agent_step=4000,  # 75% between 1000 and 5000
            l2_init_loss_coef=0.5,  # This should be ignored
            annealing_config=MockAnnealingConfig(),
        )
        expected = 0.8 + 0.75 * (0.2 - 0.8)  # 0.8 + 0.75 * (-0.6) = 0.35
        assert coef == pytest.approx(expected, abs=1e-6)
