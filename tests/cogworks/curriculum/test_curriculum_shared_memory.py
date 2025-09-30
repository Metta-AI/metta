"""Integration tests for curriculum shared memory across processes."""

import multiprocessing
import time

from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressAlgorithm, LearningProgressConfig


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
            max_memory_tasks=100,
            use_shared_memory=True,
            session_id=session_id,  # Use same session ID as parent
        )

        algorithm = LearningProgressAlgorithm(num_tasks=10, hypers=config)

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
            max_memory_tasks=100,
            use_shared_memory=True,
            session_id=session_id,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=10, hypers=config)

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
            max_memory_tasks=100,
            use_shared_memory=True,
            session_id=session_id,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=10, hypers=config)

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
            max_memory_tasks=100,
            use_shared_memory=True,
            session_id=session_id,
        )
        algorithm = LearningProgressAlgorithm(num_tasks=10, hypers=config)

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
