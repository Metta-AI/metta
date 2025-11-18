"""Tests for dual-pool curriculum configuration."""

import pytest

from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig


class TestDualPoolConfig:
    """Test dual-pool configuration parameters and validation."""

    def test_single_pool_default(self):
        """Test that single-pool is the default."""
        config = LearningProgressConfig()
        assert config.use_dual_pool is False
        assert config.num_active_tasks == 1000

    def test_dual_pool_basic_config(self):
        """Test basic dual-pool configuration."""
        config = LearningProgressConfig(
            use_dual_pool=True,
            num_explore_tasks=50,
            num_exploit_tasks=200,
        )
        assert config.use_dual_pool is True
        assert config.num_explore_tasks == 50
        assert config.num_exploit_tasks == 200
        # Should override num_active_tasks
        assert config.num_active_tasks == 250

    def test_dual_pool_default_values(self):
        """Test dual-pool default parameter values."""
        config = LearningProgressConfig(use_dual_pool=True)
        assert config.num_explore_tasks == 50
        assert config.num_exploit_tasks == 200
        assert config.promotion_min_samples == 5
        assert config.explore_exploit_ratio_init == 0.5
        assert config.explore_exploit_ratio_min == 0.05
        assert config.explore_exploit_ratio_max == 0.95
        assert config.explore_exploit_ratio_alpha == 0.9
        assert config.promotion_rate_window == 1000

    def test_dual_pool_validation_positive_pool_sizes(self):
        """Test that pool sizes must be positive."""
        with pytest.raises(ValueError, match="num_explore_tasks must be positive"):
            LearningProgressConfig(
                use_dual_pool=True,
                num_explore_tasks=0,
            )

        with pytest.raises(ValueError, match="num_exploit_tasks must be positive"):
            LearningProgressConfig(
                use_dual_pool=True,
                num_exploit_tasks=-1,
            )

    def test_dual_pool_validation_promotion_samples(self):
        """Test that promotion_min_samples must be positive."""
        with pytest.raises(ValueError, match="promotion_min_samples must be positive"):
            LearningProgressConfig(
                use_dual_pool=True,
                promotion_min_samples=0,
            )

    def test_dual_pool_validation_eer_bounds(self):
        """Test EER bounds validation."""
        # Min must be in [0, 1]
        with pytest.raises(ValueError, match="explore_exploit_ratio_min must be in"):
            LearningProgressConfig(
                use_dual_pool=True,
                explore_exploit_ratio_min=-0.1,
            )

        with pytest.raises(ValueError, match="explore_exploit_ratio_min must be in"):
            LearningProgressConfig(
                use_dual_pool=True,
                explore_exploit_ratio_min=1.1,
            )

        # Max must be in [0, 1]
        with pytest.raises(ValueError, match="explore_exploit_ratio_max must be in"):
            LearningProgressConfig(
                use_dual_pool=True,
                explore_exploit_ratio_max=1.5,
            )

        # Min must be < max
        with pytest.raises(ValueError, match="must be < explore_exploit_ratio_max"):
            LearningProgressConfig(
                use_dual_pool=True,
                explore_exploit_ratio_min=0.9,
                explore_exploit_ratio_max=0.1,
            )

        # Init must be in [0, 1]
        with pytest.raises(ValueError, match="explore_exploit_ratio_init must be in"):
            LearningProgressConfig(
                use_dual_pool=True,
                explore_exploit_ratio_init=1.5,
            )

    def test_dual_pool_validation_eer_alpha(self):
        """Test EER alpha validation."""
        # Alpha must be in (0, 1) - exclusive
        with pytest.raises(ValueError, match="explore_exploit_ratio_alpha must be in"):
            LearningProgressConfig(
                use_dual_pool=True,
                explore_exploit_ratio_alpha=0.0,
            )

        with pytest.raises(ValueError, match="explore_exploit_ratio_alpha must be in"):
            LearningProgressConfig(
                use_dual_pool=True,
                explore_exploit_ratio_alpha=1.0,
            )

        with pytest.raises(ValueError, match="explore_exploit_ratio_alpha must be in"):
            LearningProgressConfig(
                use_dual_pool=True,
                explore_exploit_ratio_alpha=-0.1,
            )

    def test_dual_pool_validation_window_size(self):
        """Test promotion rate window validation."""
        with pytest.raises(ValueError, match="promotion_rate_window must be positive"):
            LearningProgressConfig(
                use_dual_pool=True,
                promotion_rate_window=0,
            )

    def test_dual_pool_custom_pool_sizes(self):
        """Test custom pool sizes work correctly."""
        config = LearningProgressConfig(
            use_dual_pool=True,
            num_explore_tasks=100,
            num_exploit_tasks=500,
        )
        assert config.num_explore_tasks == 100
        assert config.num_exploit_tasks == 500
        assert config.num_active_tasks == 600

    def test_dual_pool_custom_eer_params(self):
        """Test custom EER parameters."""
        config = LearningProgressConfig(
            use_dual_pool=True,
            explore_exploit_ratio_init=0.7,
            explore_exploit_ratio_min=0.1,
            explore_exploit_ratio_max=0.9,
            explore_exploit_ratio_alpha=0.95,
            promotion_rate_window=500,
        )
        assert config.explore_exploit_ratio_init == 0.7
        assert config.explore_exploit_ratio_min == 0.1
        assert config.explore_exploit_ratio_max == 0.9
        assert config.explore_exploit_ratio_alpha == 0.95
        assert config.promotion_rate_window == 500

    def test_single_pool_ignores_dual_pool_params(self):
        """Test that dual-pool params don't affect single-pool mode."""
        config = LearningProgressConfig(
            use_dual_pool=False,
            num_active_tasks=1000,
            # These should be ignored but not cause errors
            num_explore_tasks=999,
            num_exploit_tasks=999,
        )
        assert config.use_dual_pool is False
        assert config.num_active_tasks == 1000  # Not overridden when dual_pool=False

    def test_dual_pool_with_shared_memory(self):
        """Test dual-pool works with shared memory enabled."""
        config = LearningProgressConfig(
            use_dual_pool=True,
            use_shared_memory=True,
        )
        assert config.use_shared_memory is True
        assert config.session_id is not None
        assert config.session_id.startswith("lp_")

    def test_dual_pool_session_id_generation(self):
        """Test that session IDs are generated when not provided."""
        config1 = LearningProgressConfig(
            use_dual_pool=True,
            use_shared_memory=True,
        )
        config2 = LearningProgressConfig(
            use_dual_pool=True,
            use_shared_memory=True,
        )

        # Each config should get a unique session ID
        assert config1.session_id != config2.session_id
        assert config1.session_id.startswith("lp_")
        assert config2.session_id.startswith("lp_")
