"""Test stats aggregation for dictionary-based curriculum stats.

This test verifies that count-based stats (like per_label_samples)
are summed across rollout steps, while rate-based stats are averaged.
"""

import pytest

from metta.rl.stats import process_training_stats
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import Experience


def test_curriculum_count_stats_are_summed():
    """Test that curriculum count stats are summed, not averaged."""
    # Simulate stats from multiple rollout steps with per-label counts
    # After unroll_nested_dict, dictionary values become separate keys
    raw_stats = {
        # These are counts that accumulated across rollout steps
        # e.g., rollout 1 had 2 completions, rollout 2 had 1, rollout 3 had 3
        "env_curriculum_stats/per_label_samples/lonely_heart": [2, 1, 3],
        "env_curriculum_stats/per_label_samples/heart_chorus": [1, 2],
        "env_curriculum_stats/per_label_evictions/lonely_heart": [1, 0, 1],
        # Regular stats should still be averaged
        "agent/rewards": [0.5, 0.6, 0.7],
    }

    losses_stats = {}
    experience = MockExperience()
    trainer_config = TrainerConfig()

    result = process_training_stats(raw_stats, losses_stats, experience, trainer_config)
    mean_stats = result["mean_stats"]

    # Count stats should be summed
    assert mean_stats["env_curriculum_stats/per_label_samples/lonely_heart"] == 6  # 2+1+3
    assert mean_stats["env_curriculum_stats/per_label_samples/heart_chorus"] == 3  # 1+2
    assert mean_stats["env_curriculum_stats/per_label_evictions/lonely_heart"] == 2  # 1+0+1

    # Regular stats should be averaged
    assert mean_stats["agent/rewards"] == pytest.approx(0.6)  # (0.5+0.6+0.7)/3


def test_tracked_task_completions_are_summed():
    """Test that tracked task completion counts are summed."""
    raw_stats = {
        "env_curriculum_stats/tracked_task_completions/task_0": [3, 2, 1],
        "env_curriculum_stats/tracked_task_completions/task_1": [1, 1],
    }

    losses_stats = {}
    experience = MockExperience()
    trainer_config = TrainerConfig()

    result = process_training_stats(raw_stats, losses_stats, experience, trainer_config)
    mean_stats = result["mean_stats"]

    # Task completions should be summed
    assert mean_stats["env_curriculum_stats/tracked_task_completions/task_0"] == 6  # 3+2+1
    assert mean_stats["env_curriculum_stats/tracked_task_completions/task_1"] == 2  # 1+1


def test_non_list_stats_pass_through():
    """Test that non-list stats are passed through unchanged."""
    raw_stats = {
        "env_curriculum_stats/per_label_samples/lonely_heart": 5,  # Already a scalar
        "agent/some_metric": "not_a_number",  # Non-numeric
    }

    losses_stats = {}
    experience = MockExperience()
    trainer_config = TrainerConfig()

    result = process_training_stats(raw_stats, losses_stats, experience, trainer_config)
    mean_stats = result["mean_stats"]

    # Scalar should pass through
    assert mean_stats["env_curriculum_stats/per_label_samples/lonely_heart"] == 5
    # Non-numeric should pass through
    assert mean_stats["agent/some_metric"] == "not_a_number"


class MockExperience(Experience):
    """Mock Experience for testing."""

    def __init__(self):
        # Don't call super().__init__() to avoid complex setup
        pass

    def stats(self):
        return {}
