"""Tests for metta.rl.profile module."""

import json

from metta.rl.profile import Profile


class TestProfile:
    """Test the Profile class and its methods."""

    def test_profile_initialization(self):
        """Test Profile initialization."""
        profile = Profile()
        assert profile.SPS == 0
        assert profile.uptime == 0
        assert profile.remaining == 0

    def test_profile_numeric_attributes(self):
        """Test that Profile attributes return numeric values."""
        profile = Profile()

        # Test timing attributes return numeric values (not strings)
        timing_attrs = [
            "eval_time",
            "train_time",
            "env_time",
            "eval_forward_time",
            "eval_misc_time",
            "train_forward_time",
            "learn_time",
            "train_misc_time",
        ]

        for attr in timing_attrs:
            value = getattr(profile, attr)
            assert isinstance(value, (int, float)), f"{attr} should return numeric value, got {type(value)}"
            assert value == 0  # Should be 0 initially

    def test_profile_update_stats(self):
        """Test the update_stats method."""
        profile = Profile()
        profile.start_time = 100.0

        # Mock time to be 110 seconds later
        import time

        original_time = time.time
        time.time = lambda: 110.0

        try:
            profile.update_stats(global_step=1000, total_timesteps=10000)

            # Check calculations
            assert profile.uptime == 10.0  # 110 - 100
            assert profile.SPS == 100.0  # 1000 / 10
            assert profile.remaining == 90.0  # (10000 - 1000) / 100
        finally:
            time.time = original_time

    def test_profile_context_managers(self):
        """Test that profile attributes can be used as context managers."""
        profile = Profile()

        # These should return ProfileTimer objects that can be used with 'with'
        context_attrs = ["env", "eval_forward", "eval_misc", "train_forward", "learn", "train_misc"]

        for attr in context_attrs:
            ctx_manager = getattr(profile, attr)
            # Just verify it has __enter__ and __exit__ methods
            assert hasattr(ctx_manager, "__enter__"), f"{attr} should have __enter__ method"
            assert hasattr(ctx_manager, "__exit__"), f"{attr} should have __exit__ method"

    def test_profile_epoch_time_property(self):
        """Test the epoch_time property."""
        profile = Profile()

        # The epoch_time property should be the sum of train_time and eval_time
        # Since we can't easily mock internal state, we just verify it returns a number
        assert isinstance(profile.epoch_time, (int, float))
        assert profile.epoch_time >= 0

    def test_trainer_performance_metrics_numeric(self):
        """Test that mimics how trainer.py gets numeric performance metrics.

        This test verifies that the fix in trainer._process_stats() works correctly
        by ensuring performance metrics are numeric values suitable for WandB.

        Note: agent_raw/* metrics are no longer created in mettagrid_env.py to prevent
        thousands of per-agent metrics from overwhelming WandB.
        """
        profile = Profile()
        profile.uptime = 100.0  # Simulate some uptime

        # This is what we do in trainer._process_stats()
        performance = {}
        for metric in [
            "eval_time",
            "env_time",
            "eval_forward_time",
            "eval_misc_time",
            "train_time",
            "train_forward_time",
            "learn_time",
            "train_misc_time",
        ]:
            # Get the raw numeric value (elapsed time in seconds)
            value = getattr(profile, metric)
            # Convert to percentage of uptime for consistency with previous behavior
            if profile.uptime > 0:
                performance[metric] = 100 * value / profile.uptime
            else:
                performance[metric] = 0.0

        # Add other numeric metrics
        performance["SPS"] = profile.SPS
        performance["uptime"] = profile.uptime
        performance["remaining"] = profile.remaining

        # Verify all values are numeric (not strings)
        for key, value in performance.items():
            assert isinstance(value, (int, float)), (
                f"Performance metric '{key}' should be numeric, got {type(value).__name__}: {value!r}"
            )

        # Verify we can pass these to wandb.log (which expects numeric values)
        # In real code this would be: wandb_run.log(performance)
        # Here we just verify the values are JSON-serializable numbers
        json_str = json.dumps(performance)
        loaded = json.loads(json_str)
        for key, value in loaded.items():
            assert isinstance(value, (int, float)), f"After JSON round-trip, {key} is not numeric"
