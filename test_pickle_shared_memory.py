#!/usr/bin/env python3
"""Test that SharedMemoryBackend and TaskTracker properly survive pickling."""

import pickle

from metta.cogworks.curriculum.task_tracker import TaskTracker


def test_task_tracker_pickle():
    """Test that TaskTracker can be pickled and unpickled with shared memory."""
    print("Creating TaskTracker with shared memory...")
    tracker1 = TaskTracker(
        max_memory_tasks=10,
        ema_alpha=0.1,
        session_id="test_session_123",
        use_shared_memory=True,
    )

    # Add a task
    print("Tracking task creation...")
    tracker1.track_task_creation(task_id=42, success_threshold=0.5)

    # Update task performance
    print("Updating task performance...")
    tracker1.update_task_performance(task_id=42, score=0.75)

    # Get stats
    stats1 = tracker1.get_task_stats(42)
    print(f"Stats before pickle: {stats1}")

    # Pickle and unpickle
    print("\nPickling TaskTracker...")
    pickled = pickle.dumps(tracker1)
    print(f"Pickled size: {len(pickled)} bytes")

    print("Unpickling TaskTracker...")
    tracker2 = pickle.loads(pickled)

    # Check that stats are preserved
    stats2 = tracker2.get_task_stats(42)
    print(f"Stats after pickle: {stats2}")

    # Verify values match
    assert stats1["completion_count"] == stats2["completion_count"], "Completion count mismatch!"
    assert abs(stats1["reward_ema"] - stats2["reward_ema"]) < 0.001, "Reward EMA mismatch!"
    print("\n✅ Pickle test PASSED - shared memory connection survived!")

    # Update from second instance
    print("\nUpdating from unpickled instance...")
    tracker2.update_task_performance(task_id=42, score=0.85)

    # Check that first instance sees the update
    stats1_updated = tracker1.get_task_stats(42)
    stats2_updated = tracker2.get_task_stats(42)
    print(f"Stats from tracker1: {stats1_updated}")
    print(f"Stats from tracker2: {stats2_updated}")

    assert stats1_updated["completion_count"] == 2, f"Expected 2 completions, got {stats1_updated['completion_count']}"
    assert stats2_updated["completion_count"] == 2, f"Expected 2 completions, got {stats2_updated['completion_count']}"
    print("✅ Shared memory test PASSED - both instances share the same memory!")

    # Cleanup
    tracker1.cleanup_shared_memory()
    print("\n🎉 ALL TESTS PASSED!")


if __name__ == "__main__":
    test_task_tracker_pickle()
