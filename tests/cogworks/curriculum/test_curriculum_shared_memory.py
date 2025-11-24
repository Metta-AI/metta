"""Integration tests for curriculum shared memory across processes."""

import multiprocessing
import time

from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressAlgorithm, LearningProgressConfig
from metta.cogworks.curriculum.stats import NullStatsLogger


def child_process_reader(
    unique_task_id: int, expected_score: float, result_queue: multiprocessing.Queue, session_id: str
) -> None:
    """Child process that reads from shared memory and verifies the value.

    Args:
        unique_task_id: The task ID to look up in shared memory
        expected_score: The expected score value to verify
        result_queue: Queue to return the verification result
        session_id: Shared memory session ID to connect to
    """
    try:
        # Create algorithm in child process - it should connect to existing shared memory
        config = LearningProgressConfig(
            ema_timescale=0.001,
            num_active_tasks=100,
            use_shared_memory=True,
            session_id=session_id,  # Use same session ID as parent
        )

        algorithm = LearningProgressAlgorithm(num_tasks=10, stats_logger=NullStatsLogger(), hypers=config)

        # Wait briefly to ensure parent has written the data
        time.sleep(0.1)

        # Read task stats from shared memory
        task_stats = algorithm.task_tracker.get_task_stats(unique_task_id)

        if task_stats is None:
            result_queue.put({"success": False, "error": f"Task {unique_task_id} not found in shared memory"})
            return

        # Verify the score matches what parent wrote
        last_score = task_stats.get("last_score", None)
        if last_score is None:
            result_queue.put({"success": False, "error": "last_score not found in task stats"})
            return

        if abs(last_score - expected_score) < 0.0001:  # Float comparison with tolerance
            result_queue.put(
                {
                    "success": True,
                    "last_score": last_score,
                    "expected_score": expected_score,
                    "task_stats": task_stats,
                }
            )
        else:
            result_queue.put(
                {
                    "success": False,
                    "error": f"Score mismatch: expected {expected_score}, got {last_score}",
                }
            )

    except Exception as e:
        import traceback

        result_queue.put({"success": False, "error": f"Child process error: {str(e)}\n{traceback.format_exc()}"})


class TestSharedMemoryIntegration:
    """Integration tests verifying shared memory works across processes."""

    def test_shared_memory_cross_process_communication(self):
        """Test that parent process writes to shared memory and child process reads it correctly."""
        # Use fork method for multiprocessing to enable proper shared memory access
        ctx = multiprocessing.get_context("fork")

        # Create unique task ID and score for this test
        unique_task_id = 123456
        unique_score = 0.7890123

        # Create unique session ID for this test with timestamp
        import time
        import uuid

        session_id = f"t{int(time.time() * 1000) % 100000}_{uuid.uuid4().hex[:4]}"

        # Create algorithm with shared memory enabled in parent process
        config = LearningProgressConfig(
            ema_timescale=0.001,
            num_active_tasks=100,
            use_shared_memory=True,
            session_id=session_id,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=10, stats_logger=NullStatsLogger(), hypers=config)

        # Create and track a task in parent process
        algorithm.task_tracker.track_task_creation(unique_task_id)

        # Update task performance with unique value in parent process
        algorithm.update_task_performance(unique_task_id, unique_score)

        # Verify parent can read the value
        parent_stats = algorithm.task_tracker.get_task_stats(unique_task_id)
        assert parent_stats is not None, "Parent should be able to read task stats"
        assert abs(parent_stats["last_score"] - unique_score) < 0.0001, "Parent should read correct score"

        # Create queue for child process to return results
        result_queue = ctx.Queue()

        # Spawn child process to read from shared memory
        child_process = ctx.Process(
            target=child_process_reader,
            args=(unique_task_id, unique_score, result_queue, session_id),
        )
        child_process.start()
        child_process.join(timeout=5.0)  # Wait max 5 seconds

        # Check that child process completed
        assert not child_process.is_alive(), "Child process should have completed"
        assert child_process.exitcode == 0, "Child process should exit successfully"

        # Get result from child process
        assert not result_queue.empty(), "Child process should return a result"
        result = result_queue.get()

        # Verify child process successfully read the correct value
        assert result["success"], f"Child process verification failed: {result.get('error', 'Unknown error')}"
        assert abs(result["last_score"] - unique_score) < 0.0001, (
            f"Child read incorrect score: expected {unique_score}, got {result['last_score']}"
        )

        # Cleanup shared memory
        algorithm.cleanup_shared_memory()

    def test_shared_memory_multiple_tasks_cross_process(self):
        """Test that multiple tasks can be written by parent and read by child."""
        # Use fork method for multiprocessing to enable proper shared memory access
        ctx = multiprocessing.get_context("fork")

        # Create unique session ID with timestamp
        import time
        import uuid

        session_id = f"t{int(time.time() * 1000) % 100000}_{uuid.uuid4().hex[:4]}"

        # Create multiple unique task IDs and scores
        task_data = [
            (111111, 0.111),
            (222222, 0.222),
            (333333, 0.333),
            (444444, 0.444),
        ]

        # Create algorithm with shared memory in parent
        config = LearningProgressConfig(
            ema_timescale=0.001,
            num_active_tasks=100,
            use_shared_memory=True,
            session_id=session_id,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=10, stats_logger=NullStatsLogger(), hypers=config)

        # Write multiple tasks in parent process
        for task_id, score in task_data:
            algorithm.task_tracker.track_task_creation(task_id)
            algorithm.update_task_performance(task_id, score)

        # Verify all tasks in parent
        for task_id, score in task_data:
            stats = algorithm.task_tracker.get_task_stats(task_id)
            assert stats is not None
            assert abs(stats["last_score"] - score) < 0.0001

        # Spawn child processes to verify each task
        result_queue = ctx.Queue()
        processes = []

        for task_id, score in task_data:
            p = ctx.Process(
                target=child_process_reader,
                args=(task_id, score, result_queue, session_id),
            )
            p.start()
            processes.append(p)

        # Wait for all child processes
        for p in processes:
            p.join(timeout=5.0)
            assert not p.is_alive(), "Child process should complete"
            assert p.exitcode == 0, "Child process should exit successfully"

        # Verify all results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        assert len(results) == len(task_data), f"Expected {len(task_data)} results, got {len(results)}"

        for result in results:
            assert result["success"], f"Child process failed: {result.get('error', 'Unknown')}"

        # Cleanup
        algorithm.cleanup_shared_memory()

    def test_shared_memory_persistence_after_parent_update(self):
        """Test that child can read updated values after parent modifies them."""
        # Use fork method for multiprocessing to enable proper shared memory access
        ctx = multiprocessing.get_context("fork")

        # Create unique session ID with timestamp
        import time
        import uuid

        session_id = f"t{int(time.time() * 1000) % 100000}_{uuid.uuid4().hex[:4]}"

        task_id = 555555
        initial_score = 0.5
        updated_score = 0.9

        # Create algorithm in parent
        config = LearningProgressConfig(
            ema_timescale=0.001,
            num_active_tasks=100,
            use_shared_memory=True,
            session_id=session_id,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=10, stats_logger=NullStatsLogger(), hypers=config)

        # Write initial value
        algorithm.task_tracker.track_task_creation(task_id)
        algorithm.update_task_performance(task_id, initial_score)

        # Verify initial value
        stats = algorithm.task_tracker.get_task_stats(task_id)
        assert stats is not None
        assert abs(stats["last_score"] - initial_score) < 0.0001

        # Update with new value
        algorithm.update_task_performance(task_id, updated_score)

        # Verify updated value in parent
        stats = algorithm.task_tracker.get_task_stats(task_id)
        assert abs(stats["last_score"] - updated_score) < 0.0001

        # Child process should read the updated value
        result_queue = ctx.Queue()
        child_process = ctx.Process(
            target=child_process_reader,
            args=(task_id, updated_score, result_queue, session_id),
        )
        child_process.start()
        child_process.join(timeout=5.0)

        assert not child_process.is_alive()
        assert child_process.exitcode == 0

        result = result_queue.get()
        assert result["success"], f"Child failed to read updated value: {result.get('error')}"
        assert abs(result["last_score"] - updated_score) < 0.0001

        # Cleanup
        algorithm.cleanup_shared_memory()

    def test_slot_reuse_after_eviction(self):
        """Test that memory slots are reused after tasks are evicted.

        This is a critical test for the bug fix where _next_free_index wasn't being
        updated when tasks were evicted, causing memory exhaustion even with available slots.
        """
        import time
        import uuid

        # Create unique session ID
        session_id = f"t{int(time.time() * 1000) % 100000}_{uuid.uuid4().hex[:4]}"

        # Use small max_memory_tasks to make test faster
        max_tasks = 20
        config = LearningProgressConfig(
            ema_timescale=0.001,
            num_active_tasks=max_tasks,
            use_shared_memory=True,
            session_id=session_id,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=10, stats_logger=NullStatsLogger(), hypers=config)

        # Phase 1: Fill up all slots
        initial_task_ids = list(range(1000, 1000 + max_tasks))
        for task_id in initial_task_ids:
            algorithm.task_tracker.track_task_creation(task_id)
            algorithm.update_task_performance(task_id, 0.5)

        # Verify all tasks are tracked
        tracked_tasks = algorithm.task_tracker.get_all_tracked_tasks()
        assert len(tracked_tasks) == max_tasks, f"Expected {max_tasks} tasks, got {len(tracked_tasks)}"

        # Phase 2: Evict half of the tasks
        tasks_to_evict = initial_task_ids[: max_tasks // 2]
        for task_id in tasks_to_evict:
            algorithm.task_tracker.remove_task(task_id)

        # Verify tasks were evicted
        tracked_after_eviction = algorithm.task_tracker.get_all_tracked_tasks()
        assert len(tracked_after_eviction) == max_tasks // 2, (
            f"Expected {max_tasks // 2} tasks after eviction, got {len(tracked_after_eviction)}"
        )

        # Phase 3: Create new tasks (should reuse freed slots)
        new_task_ids = list(range(2000, 2000 + max_tasks // 2))
        for task_id in new_task_ids:
            algorithm.task_tracker.track_task_creation(task_id)
            algorithm.update_task_performance(task_id, 0.7)

        # Verify new tasks were created successfully
        tracked_after_refill = algorithm.task_tracker.get_all_tracked_tasks()
        assert len(tracked_after_refill) == max_tasks, (
            f"Expected {max_tasks} tasks after refill, got {len(tracked_after_refill)}"
        )

        # Verify new tasks are actually tracked
        for task_id in new_task_ids:
            stats = algorithm.task_tracker.get_task_stats(task_id)
            assert stats is not None, f"New task {task_id} should be tracked"
            assert abs(stats["last_score"] - 0.7) < 0.0001, f"New task {task_id} should have correct score"

        # Verify old evicted tasks are not tracked
        for task_id in tasks_to_evict:
            stats = algorithm.task_tracker.get_task_stats(task_id)
            assert stats is None, f"Evicted task {task_id} should not be tracked"

        # Phase 4: Test the full cycle - evict all and refill completely
        # This tests the scenario that caused the original bug
        for task_id in tracked_after_refill:
            algorithm.task_tracker.remove_task(task_id)

        # Verify all tasks evicted
        tracked_after_full_eviction = algorithm.task_tracker.get_all_tracked_tasks()
        assert len(tracked_after_full_eviction) == 0, "All tasks should be evicted"

        # Create full set of new tasks
        final_task_ids = list(range(3000, 3000 + max_tasks))
        for task_id in final_task_ids:
            algorithm.task_tracker.track_task_creation(task_id)
            algorithm.update_task_performance(task_id, 0.9)

        # Verify all new tasks were created (this would fail with the bug)
        tracked_final = algorithm.task_tracker.get_all_tracked_tasks()
        assert len(tracked_final) == max_tasks, (
            f"Expected {max_tasks} tasks after full refill, got {len(tracked_final)}. "
            "This indicates memory slots are not being reused!"
        )

        # Verify all final tasks have correct data
        for task_id in final_task_ids:
            stats = algorithm.task_tracker.get_task_stats(task_id)
            assert stats is not None, f"Final task {task_id} should be tracked"
            assert abs(stats["last_score"] - 0.9) < 0.0001, (
                f"Final task {task_id} should have score 0.9, got {stats['last_score']}"
            )

        # Cleanup
        algorithm.cleanup_shared_memory()

    def test_slot_reuse_pattern_matches_expectations(self):
        """Test that slots are reused in the expected order (lowest index first)."""
        import time
        import uuid

        # Create unique session ID
        session_id = f"t{int(time.time() * 1000) % 100000}_{uuid.uuid4().hex[:4]}"

        max_tasks = 10
        config = LearningProgressConfig(
            ema_timescale=0.001,
            num_active_tasks=max_tasks,
            use_shared_memory=True,
            session_id=session_id,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=10, stats_logger=NullStatsLogger(), hypers=config)

        # Create tasks at specific indices
        task_ids = list(range(100, 110))  # 10 tasks
        for task_id in task_ids:
            algorithm.task_tracker.track_task_creation(task_id)

        # Get the internal index mapping (for verification)
        task_id_to_index = algorithm.task_tracker._task_id_to_index.copy()

        # Evict tasks at indices 2, 5, 7
        tasks_at_indices = {idx: tid for tid, idx in task_id_to_index.items()}
        evict_indices = [2, 5, 7]
        evicted_task_ids = [tasks_at_indices[idx] for idx in evict_indices]

        for task_id in evicted_task_ids:
            algorithm.task_tracker.remove_task(task_id)

        # Check _next_free_index points to lowest freed slot
        assert algorithm.task_tracker._next_free_index == min(evict_indices), (
            f"Expected _next_free_index to be {min(evict_indices)}, got {algorithm.task_tracker._next_free_index}"
        )

        # Create a new task - should use index 2 (lowest freed)
        new_task_id = 200
        algorithm.task_tracker.track_task_creation(new_task_id)
        new_task_index = algorithm.task_tracker._task_id_to_index[new_task_id]
        assert new_task_index == 2, f"New task should use index 2, got {new_task_index}"

        # _next_free_index should now point to next freed slot (5)
        assert algorithm.task_tracker._next_free_index == 5, (
            f"Expected _next_free_index to be 5, got {algorithm.task_tracker._next_free_index}"
        )

        # Create another task - should use index 5
        new_task_id_2 = 201
        algorithm.task_tracker.track_task_creation(new_task_id_2)
        new_task_index_2 = algorithm.task_tracker._task_id_to_index[new_task_id_2]
        assert new_task_index_2 == 5, f"Second new task should use index 5, got {new_task_index_2}"

        # _next_free_index should now point to index 7
        assert algorithm.task_tracker._next_free_index == 7, (
            f"Expected _next_free_index to be 7, got {algorithm.task_tracker._next_free_index}"
        )

        # Cleanup
        algorithm.cleanup_shared_memory()
