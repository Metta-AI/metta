"""Unit tests for CurriculumClient."""

import time
from unittest.mock import patch

import numpy as np
import pytest

from metta.rl.curriculum import CurriculumClient, CurriculumManager, Task, TaskState


class TestCurriculumClient:
    """Test suite for CurriculumClient."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CurriculumManager(pool_size=100, min_runs=5, name="test_client")
        self.curriculum_name = self.manager.get_shared_memory_names()

    def teardown_method(self):
        """Clean up test fixtures."""
        self.manager.cleanup()

    def _create_client(self, **kwargs):
        """Helper to create a CurriculumClient with defaults."""
        defaults = {
            "curriculum_name": self.curriculum_name,
            "pool_size": 100,
            "num_samples": 16,
            "min_runs": 5,
        }
        defaults.update(kwargs)
        return CurriculumClient(**defaults)

    def test_initialization(self):
        """Test CurriculumClient initialization."""
        client = self._create_client(
            num_samples=20, selection_strategy="epsilon_greedy", epsilon=0.2, temperature=1.5, ucb_c=3.0
        )

        assert client.pool_size == 100
        assert client.num_samples == 20
        assert client.min_runs == 5
        assert client.selection_strategy == "epsilon_greedy"
        assert client.epsilon == 0.2
        assert client.temperature == 1.5
        assert client.ucb_c == 3.0

        # Clean up
        client.cleanup()

    def test_num_samples_capped_by_pool_size(self):
        """Test that num_samples is capped by pool_size."""
        client = self._create_client(num_samples=200)  # Larger than pool_size

        assert client.num_samples == 100  # Should be capped

        # Clean up
        client.cleanup()

    def test_get_task_basic(self):
        """Test basic task retrieval."""
        client = self._create_client(pool_size=100, num_samples=10, min_runs=5)

        task = client.get_task()

        assert isinstance(task, Task)
        assert 0 <= task.task_id < 2**31
        assert hasattr(task, "_slot_id")
        assert hasattr(task, "_client")

    def test_task_replacement(self):
        """Test that lowest scoring task is replaced when eligible."""
        # Set up some tasks with known scores
        for i in range(10):
            task_state = TaskState(
                task_id=1000 + i,
                score=i * 0.1,  # Scores: 0.0, 0.1, 0.2, ..., 0.9
                num_runs=10,  # All eligible for replacement
                last_update=time.time(),
                reward_mean=0.5,
                reward_var=0.01,
            )
            self.manager._set_task_state(i, task_state)

        client = self._create_client(pool_size=100, num_samples=5, min_runs=5)

        # Get a task (should trigger replacement of lowest score)
        task = client.get_task()

        # Check that task with ID 1000 (lowest score) was replaced
        replaced_task = self.manager._get_task_state(0)
        assert replaced_task.task_id != 1000
        assert replaced_task.num_runs == 0
        assert replaced_task.score == 0.0

    def test_no_replacement_when_insufficient_runs(self):
        """Test that tasks with < min_runs are not replaced."""
        # Set up tasks with insufficient runs
        for i in range(10):
            task_state = TaskState(
                task_id=2000 + i,
                score=i * 0.1,
                num_runs=3,  # Less than min_runs
                last_update=time.time(),
                reward_mean=0.5,
                reward_var=0.01,
            )
            self.manager._set_task_state(i, task_state)

        client = self._create_client(pool_size=100, num_samples=5, min_runs=5)

        # Get a task (should not trigger replacement)
        task = client.get_task()

        # Check that no tasks were replaced
        for i in range(10):
            task_state = self.manager._get_task_state(i)
            assert task_state.task_id == 2000 + i

    def test_epsilon_greedy_selection(self):
        """Test epsilon-greedy selection strategy."""
        # Set up tasks with distinct scores
        for i in range(10):
            task_state = TaskState(task_id=3000 + i, score=i * 0.1, num_runs=1, last_update=time.time())
            self.manager._set_task_state(i, task_state)

        # Test with epsilon=0 (always exploit)
        client = self._create_client(
            pool_size=100, num_samples=10, min_runs=20, selection_strategy="epsilon_greedy", epsilon=0.0
        )

        # Should always select the lowest score task
        selected_scores = []
        for _ in range(20):
            task = client.get_task()
            task_state = self.manager._get_task_state(task._slot_id)
            selected_scores.append(task_state.score)

        # All selected scores should be from the lower end
        assert max(selected_scores) < 0.5

    def test_softmax_selection(self):
        """Test softmax selection strategy."""
        # Set up tasks with distinct scores
        for i in range(20):
            task_state = TaskState(
                task_id=4000 + i,
                score=i * 0.05,  # Scores from 0.0 to 0.95
                num_runs=1,
                last_update=time.time(),
            )
            self.manager._set_task_state(i, task_state)

        client = self._create_client(
            pool_size=100,
            num_samples=20,
            min_runs=20,
            selection_strategy="softmax",
            temperature=0.5,  # Lower temperature = more exploitation
        )

        # Sample many tasks and check distribution
        selected_scores = []
        for _ in range(100):
            task = client.get_task()
            task_state = self.manager._get_task_state(task._slot_id)
            selected_scores.append(task_state.score)

        # Lower score tasks should be selected more often
        avg_score = np.mean(selected_scores)
        assert avg_score < 0.5  # Should favor lower scores

    def test_ucb_selection(self):
        """Test UCB (Upper Confidence Bound) selection strategy."""
        # Set up tasks with different run counts
        for i in range(10):
            task_state = TaskState(
                task_id=5000 + i,
                score=0.5,  # Same score for all
                num_runs=i,  # Different run counts
                last_update=time.time(),
            )
            self.manager._set_task_state(i, task_state)

        client = self._create_client(pool_size=100, num_samples=10, min_runs=20, selection_strategy="ucb", ucb_c=2.0)

        # UCB should prioritize tasks with fewer runs (more exploration)
        selected_runs = []
        for _ in range(20):
            task = client.get_task()
            task_state = self.manager._get_task_state(task._slot_id)
            selected_runs.append(task_state.num_runs)

        # Should mostly select tasks with 0 runs
        assert selected_runs.count(0) > 10

    def test_complete_task(self):
        """Test task completion and score update."""
        # Set up a task
        task_state = TaskState(task_id=6000, score=0.5, num_runs=5, last_update=time.time())
        self.manager._set_task_state(0, task_state)

        client = self._create_client()

        # Complete the task
        client.complete_task(slot_id=0, task_id=6000, reward_mean=0.8, reward_var=0.02)

        # Check updated state
        updated_state = self.manager._get_task_state(0)
        assert updated_state.num_runs == 6
        assert updated_state.reward_mean == 0.8
        assert updated_state.reward_var == 0.02
        assert 0.0 <= updated_state.score <= 1.0

    def test_complete_task_mismatch(self):
        """Test that completing a mismatched task is handled correctly."""
        # Set up a task
        task_state = TaskState(task_id=7000, score=0.5, num_runs=5, last_update=time.time())
        self.manager._set_task_state(0, task_state)

        client = self._create_client()

        # Try to complete a different task ID
        with patch("metta.rl.curriculum.client.logger") as mock_logger:
            client.complete_task(slot_id=0, task_id=7001, reward_mean=0.8, reward_var=0.02)

            # Should log a warning
            mock_logger.warning.assert_called_once()

        # State should not be updated
        unchanged_state = self.manager._get_task_state(0)
        assert unchanged_state.num_runs == 5

    def test_compute_score(self):
        """Test score computation logic."""
        client = self._create_client(pool_size=100, num_samples=5, min_runs=5)

        # Test various scenarios
        # Easy task (high reward, low variance)
        score1 = client._compute_score(reward_mean=0.9, reward_var=0.01, num_runs=10)
        assert score1 < 0.3  # Should have low score (easy)

        # Hard task (low reward)
        score2 = client._compute_score(reward_mean=0.1, reward_var=0.01, num_runs=10)
        assert score2 > 0.5  # Should have relatively high score (hard)

        # Uncertain task (high variance)
        score3 = client._compute_score(reward_mean=0.5, reward_var=0.5, num_runs=1)
        assert 0.2 < score3 < 0.4  # Should be in lower-middle range due to low confidence

        # Test confidence effect - with high confidence, score converges to difficulty score
        score_low_runs = client._compute_score(reward_mean=0.5, reward_var=0.1, num_runs=1)
        score_high_runs = client._compute_score(reward_mean=0.5, reward_var=0.1, num_runs=100)
        # With high runs, score should be close to difficulty_score (0.5)
        assert abs(score_high_runs - 0.5) < 0.01
        # With low runs, score should be affected by uncertainty
        assert abs(score_low_runs - 0.5) > 0.1

    def test_get_stats(self):
        """Test client statistics."""
        client = self._create_client(
            pool_size=100, num_samples=10, min_runs=5, selection_strategy="epsilon_greedy", epsilon=0.15
        )

        # Get some tasks to update selection count
        for _ in range(5):
            client.get_task()

        stats = client.get_stats()

        assert stats["selection_strategy"] == "epsilon_greedy"
        assert stats["num_samples"] == 10
        assert stats["total_selections"] == 5
        assert stats["epsilon"] == 0.15
        assert stats["temperature"] is None  # Not used for epsilon_greedy
        assert stats["ucb_c"] is None  # Not used for epsilon_greedy

    def test_invalid_selection_strategy(self):
        """Test that invalid selection strategy raises error."""
        client = self._create_client(pool_size=100, num_samples=5, min_runs=5, selection_strategy="invalid_strategy")

        with pytest.raises(ValueError, match="Unknown selection strategy"):
            client.get_task()

    def test_concurrent_task_operations(self):
        """Test thread-safe task selection and completion."""
        import threading

        client = self._create_client(pool_size=100, num_samples=10, min_runs=5)

        completed_tasks = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    task = client.get_task()
                    time.sleep(0.001)  # Simulate work
                    task.complete(reward_mean=np.random.rand(), reward_var=0.1)
                    completed_tasks.append(task.task_id)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = []
        for _ in range(4):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join()

        # Check no errors occurred
        assert len(errors) == 0
        assert len(completed_tasks) == 40  # 4 threads * 10 tasks each


if __name__ == "__main__":
    pytest.main([__file__])
