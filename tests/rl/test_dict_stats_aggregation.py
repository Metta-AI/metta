"""Test that per-label count stats are properly summed (not averaged) across vectorized environments."""

from metta.rl.stats import accumulate_rollout_stats, process_training_stats


def test_per_label_counts_are_summed_not_averaged():
    """Test that per-label counts from multiple envs are summed, not averaged."""
    # Simulate info dicts from 3 different vectorized environments
    # After unroll_nested_dict, these become flat keys
    raw_infos = [
        {"env_curriculum/per_label_samples_this_epoch": {"dense_large": 1, "random": 0}},
        {"env_curriculum/per_label_samples_this_epoch": {"dense_large": 0, "random": 1}},
        {"env_curriculum/per_label_samples_this_epoch": {"dense_large": 1, "random": 1}},
    ]

    stats = {}
    accumulate_rollout_stats(raw_infos, stats)

    # After unrolling, the keys should be flattened
    assert "env_curriculum/per_label_samples_this_epoch/dense_large" in stats
    assert "env_curriculum/per_label_samples_this_epoch/random" in stats

    # Values should be accumulated as lists
    assert stats["env_curriculum/per_label_samples_this_epoch/dense_large"] == [1, 0, 1]
    assert stats["env_curriculum/per_label_samples_this_epoch/random"] == [0, 1, 1]

    # Now process the stats - should SUM the counts, not average them
    # Create a mock Experience object with minimal stats() method
    class MockExperience:
        def stats(self):
            return {}

    processed = process_training_stats(stats, {}, MockExperience(), None)
    mean_stats = processed["mean_stats"]

    # Verify counts are summed (not averaged)
    assert mean_stats["env_curriculum/per_label_samples_this_epoch/dense_large"] == 2  # sum: 1 + 0 + 1
    assert mean_stats["env_curriculum/per_label_samples_this_epoch/random"] == 2  # sum: 0 + 1 + 1


def test_per_label_counts_with_many_envs():
    """Test aggregation with many vectorized environments."""
    # Simulate 32 vectorized environments completing episodes
    raw_infos = []
    for i in range(32):
        # Each env samples one label per episode
        if i < 10:
            raw_infos.append({"env_curriculum/per_label_samples_this_epoch": {"dense_large": 1}})
        elif i < 20:
            raw_infos.append({"env_curriculum/per_label_samples_this_epoch": {"random": 1}})
        else:
            raw_infos.append({"env_curriculum/per_label_samples_this_epoch": {"maze_small": 1}})

    stats = {}
    accumulate_rollout_stats(raw_infos, stats)

    # Process the stats
    class MockExperience:
        def stats(self):
            return {}

    processed = process_training_stats(stats, {}, MockExperience(), None)
    mean_stats = processed["mean_stats"]

    # Should sum across all 32 environments
    assert mean_stats["env_curriculum/per_label_samples_this_epoch/dense_large"] == 10
    assert mean_stats["env_curriculum/per_label_samples_this_epoch/random"] == 10
    assert mean_stats["env_curriculum/per_label_samples_this_epoch/maze_small"] == 12


def test_non_count_metrics_are_averaged():
    """Test that non-count metrics (like rewards) are still averaged."""
    raw_infos = [
        {
            "env_curriculum/per_label_samples_this_epoch": {"dense_large": 1},
            "episode_reward": 10.0,
        },
        {
            "env_curriculum/per_label_samples_this_epoch": {"random": 1},
            "episode_reward": 20.0,
        },
        {
            "env_curriculum/per_label_samples_this_epoch": {"dense_large": 1},
            "episode_reward": 30.0,
        },
    ]

    stats = {}
    accumulate_rollout_stats(raw_infos, stats)

    # Process the stats
    class MockExperience:
        def stats(self):
            return {}

    processed = process_training_stats(stats, {}, MockExperience(), None)
    mean_stats = processed["mean_stats"]

    # Counts should be summed
    assert mean_stats["env_curriculum/per_label_samples_this_epoch/dense_large"] == 2
    assert mean_stats["env_curriculum/per_label_samples_this_epoch/random"] == 1

    # Rewards should be averaged (not summed)
    assert mean_stats["episode_reward"] == 20.0  # mean: (10 + 20 + 30) / 3
