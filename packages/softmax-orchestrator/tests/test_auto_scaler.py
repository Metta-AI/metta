import logging
import math
from unittest.mock import AsyncMock, Mock

import pytest

from softmax.orchestrator.eval_task_orchestrator import AutoScaler
from softmax.orchestrator.routes.eval_task_routes import TaskAvgRuntimeResponse, TaskCountResponse

# Query constants used by AutoScaler
UNPROCESSED_QUERY = "status = 'unprocessed' OR status = 'running'"
TASKS_CREATED_LAST_DAY_QUERY = "created_at > NOW() - INTERVAL '1 day'"
DONE_TASKS_LAST_DAY_QUERY = "status = 'done' AND created_at > NOW() - INTERVAL '1 day'"


class TestAutoScaler:
    """Test the AutoScaler class with a mocked EvalTaskClient."""

    @pytest.fixture
    def mock_task_client(self):
        """Create a mock EvalTaskClient."""
        mock_client = Mock()
        mock_client.count_tasks = AsyncMock()
        mock_client.get_avg_runtime = AsyncMock()
        # Ensure the mock returns actual values, not more mocks
        mock_client.count_tasks.return_value = TaskCountResponse(count=0)
        mock_client.get_avg_runtime.return_value = TaskAvgRuntimeResponse(avg_runtime=None)
        return mock_client

    @pytest.fixture
    def logger(self):
        """Create a test logger."""
        return logging.getLogger("test")

    @pytest.fixture
    def auto_scaler(self, mock_task_client, logger):
        """Create an AutoScaler with mocked dependencies."""
        return AutoScaler(task_client=mock_task_client, default_task_runtime_seconds=120.0)

    @pytest.mark.asyncio
    async def test_get_desired_workers_uses_historical_runtime(self, mock_task_client, logger):
        """Test scaling calculation uses historical runtime when enough data is available."""
        # Create a fresh AutoScaler instance to avoid fixture issues
        auto_scaler = AutoScaler(task_client=mock_task_client, default_task_runtime_seconds=120.0)

        # Setup the mock to return specific responses for different calls
        call_count = 0

        async def mock_count_tasks(where_clause):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call: unprocessed tasks
                return TaskCountResponse(count=5)
            elif call_count == 2:  # Second call: tasks created in last day
                return TaskCountResponse(count=100)
            elif call_count == 3:  # Third call: done tasks in last day
                return TaskCountResponse(count=25)  # > 20, so will call get_avg_runtime
            return TaskCountResponse(count=0)

        mock_task_client.count_tasks.side_effect = mock_count_tasks
        mock_task_client.get_avg_runtime.return_value = TaskAvgRuntimeResponse(avg_runtime=150.0)

        result = await auto_scaler.get_desired_workers(num_workers=2)

        # With 100 tasks/day * 150s/task = 15000s total work (using historical runtime)
        # Single worker can do 86400s/day, so need ceil(15000/86400 * 1.2) = 1 worker
        expected = math.ceil(100 * 150.0 / (60 * 60 * 24) * 1.2)
        assert result == expected
        assert result == 1

        # Verify get_avg_runtime was called since done tasks > 20
        mock_task_client.get_avg_runtime.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_desired_workers_uses_default_runtime(self, mock_task_client, logger):
        """Test scaling calculation uses default runtime when insufficient historical data."""
        # Reset the mock call counts
        mock_task_client.reset_mock()

        auto_scaler = AutoScaler(task_client=mock_task_client, default_task_runtime_seconds=120.0)

        # Setup the mock to return specific responses based on the exact queries used
        async def mock_count_tasks(where_clause):
            if where_clause == UNPROCESSED_QUERY:
                return TaskCountResponse(count=5)
            elif where_clause == TASKS_CREATED_LAST_DAY_QUERY:
                return TaskCountResponse(count=100)  # tasks created in last day
            elif where_clause == DONE_TASKS_LAST_DAY_QUERY:
                return TaskCountResponse(count=15)  # done tasks <= 20, so won't call get_avg_runtime
            return TaskCountResponse(count=0)

        mock_task_client.count_tasks.side_effect = mock_count_tasks

        result = await auto_scaler.get_desired_workers(num_workers=2)

        # With 100 tasks/day * 120s/task = 12000s total work (using default runtime)
        # Single worker can do 86400s/day, so need ceil(12000/86400 * 1.2) = 1 worker
        expected = math.ceil(100 * 120.0 / (60 * 60 * 24) * 1.2)
        assert result == expected
        assert result == 1

        # Verify get_avg_runtime was NOT called since done tasks <= 20
        mock_task_client.get_avg_runtime.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_desired_workers_with_backlog(self, mock_task_client, logger):
        """Test scaling when there's a large backlog of tasks."""
        # Reset the mock call counts
        mock_task_client.reset_mock()

        auto_scaler = AutoScaler(task_client=mock_task_client, default_task_runtime_seconds=120.0)

        # Setup the mock to return specific responses based on the exact queries used
        async def mock_count_tasks(where_clause):
            if where_clause == UNPROCESSED_QUERY:
                return TaskCountResponse(count=100)  # > 5 * 2 workers, triggers backlog
            elif where_clause == TASKS_CREATED_LAST_DAY_QUERY:
                return TaskCountResponse(count=500)  # tasks created in last day
            elif where_clause == DONE_TASKS_LAST_DAY_QUERY:
                return TaskCountResponse(count=10)  # done tasks <= 20, so won't call get_avg_runtime
            return TaskCountResponse(count=0)

        mock_task_client.count_tasks.side_effect = mock_count_tasks

        current_workers = 2
        result = await auto_scaler.get_desired_workers(num_workers=current_workers)

        # With 100 unclaimed tasks > 5 * 2 workers, should add workers for backlog
        # Extra workers = ceil(100 tasks * 120s/task / 3600s/hour) = ceil(3.33) = 4
        expected_workers = math.ceil(100 * 120.0 / 3600)
        assert result == expected_workers
        assert result == 4

        # Verify get_avg_runtime was not called since done tasks <= 20
        mock_task_client.get_avg_runtime.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_desired_workers_with_historical_runtime(self, auto_scaler, mock_task_client):
        """Test scaling using historical average runtime."""
        # Setup: enough completed tasks to use historical average
        mock_task_client.count_tasks.side_effect = [
            TaskCountResponse(count=5),  # unprocessed tasks
            TaskCountResponse(count=200),  # tasks created in last day
            TaskCountResponse(count=25),  # done tasks in last day (>= 20, use historical)
        ]
        mock_task_client.get_avg_runtime.return_value = TaskAvgRuntimeResponse(avg_runtime=300.0)

        result = await auto_scaler.get_desired_workers(num_workers=2)

        # With 200 tasks/day * 300s/task = 60000s total work
        # Single worker can do 86400s/day, so need ceil(60000/86400 * 1.2) = 1 worker
        expected = math.ceil(200 * 300.0 / (60 * 60 * 24) * 1.2)
        assert result == expected
        assert result == 1

    @pytest.mark.asyncio
    async def test_get_desired_workers_with_null_avg_runtime(self, auto_scaler, mock_task_client):
        """Test scaling when historical average runtime is null."""
        mock_task_client.count_tasks.side_effect = [
            TaskCountResponse(count=3),  # unprocessed tasks
            TaskCountResponse(count=150),  # tasks created in last day
            TaskCountResponse(count=30),  # done tasks in last day (>= 20, try historical)
        ]
        mock_task_client.get_avg_runtime.return_value = TaskAvgRuntimeResponse(avg_runtime=None)

        result = await auto_scaler.get_desired_workers(num_workers=1)

        # Should fall back to default runtime (120.0s)
        expected = math.ceil(150 * 120.0 / (60 * 60 * 24) * 1.2)
        assert result == expected
        assert result == 1

    @pytest.mark.asyncio
    async def test_compute_desired_workers_calculation(self, auto_scaler, mock_task_client):
        """Test the internal _compute_desired_workers calculation."""
        mock_task_client.count_tasks.return_value = TaskCountResponse(count=1000)

        # Test with specific average runtime
        avg_runtime = 180.0  # 3 minutes per task
        result = await auto_scaler._compute_desired_workers(avg_runtime)

        # 1000 tasks * 180s = 180000s total work per day
        # Single worker = 86400s per day
        # Need ceil(180000/86400 * 1.2) = ceil(2.5) = 3 workers
        expected = math.ceil(1000 * 180.0 / (60 * 60 * 24) * 1.2)
        assert result == expected
        assert result == 3

    @pytest.mark.asyncio
    async def test_get_avg_task_runtime_with_sufficient_data(self, auto_scaler, mock_task_client):
        """Test _get_avg_task_runtime with enough historical data."""
        mock_task_client.count_tasks.return_value = TaskCountResponse(count=50)  # > 20 done tasks
        mock_task_client.get_avg_runtime.return_value = TaskAvgRuntimeResponse(avg_runtime=250.0)

        result = await auto_scaler._get_avg_task_runtime()

        assert result == 250.0
        mock_task_client.get_avg_runtime.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_avg_task_runtime_with_insufficient_data(self, auto_scaler, mock_task_client):
        """Test _get_avg_task_runtime with insufficient historical data."""
        mock_task_client.count_tasks.return_value = TaskCountResponse(count=15)  # < 20 done tasks

        result = await auto_scaler._get_avg_task_runtime()

        # Should return default runtime
        assert result == 120.0
        mock_task_client.get_avg_runtime.assert_not_called()

    @pytest.mark.asyncio
    async def test_backlog_threshold_exactly_at_limit(self, mock_task_client, logger):
        """Test behavior when unclaimed tasks exactly equals the threshold."""
        # Reset the mock call counts
        mock_task_client.reset_mock()

        auto_scaler = AutoScaler(task_client=mock_task_client, default_task_runtime_seconds=120.0)

        current_workers = 3
        unclaimed_tasks = current_workers * 5  # Exactly at threshold

        # Setup the mock to return specific responses based on the exact queries used
        async def mock_count_tasks(where_clause):
            if where_clause == UNPROCESSED_QUERY:
                return TaskCountResponse(count=unclaimed_tasks)  # exactly at threshold (= not >)
            elif where_clause == TASKS_CREATED_LAST_DAY_QUERY:
                return TaskCountResponse(count=100)
            elif where_clause == DONE_TASKS_LAST_DAY_QUERY:
                return TaskCountResponse(count=10)  # <= 20, so won't call get_avg_runtime
            return TaskCountResponse(count=0)

        mock_task_client.count_tasks.side_effect = mock_count_tasks

        result = await auto_scaler.get_desired_workers(num_workers=current_workers)

        # Should NOT trigger backlog scaling (threshold is > not >=)
        # Should use normal scaling calculation
        expected = math.ceil(100 * 120.0 / (60 * 60 * 24) * 1.2)
        assert result == expected

        # Verify get_avg_runtime was not called since done tasks <= 20
        mock_task_client.get_avg_runtime.assert_not_called()

    @pytest.mark.asyncio
    async def test_large_backlog_scaling(self, auto_scaler, mock_task_client):
        """Test scaling with a very large backlog."""
        current_workers = 5
        unclaimed_tasks = 1000  # Very large backlog

        mock_task_client.count_tasks.side_effect = [
            TaskCountResponse(count=unclaimed_tasks),
            TaskCountResponse(count=200),  # tasks created in last day
            TaskCountResponse(count=25),  # done tasks in last day
        ]
        mock_task_client.get_avg_runtime.return_value = TaskAvgRuntimeResponse(avg_runtime=60.0)

        result = await auto_scaler.get_desired_workers(num_workers=current_workers)

        # Extra workers needed = ceil(1000 tasks * 60s/task / 3600s/hour) = ceil(16.67) = 17
        expected_total = math.ceil(1000 * 60.0 / 3600)
        assert result == expected_total
        assert result == 17

    @pytest.mark.asyncio
    async def test_zero_tasks_scenario(self, auto_scaler, mock_task_client):
        """Test behavior when there are no tasks."""
        mock_task_client.count_tasks.side_effect = [
            TaskCountResponse(count=0),  # no unprocessed tasks
            TaskCountResponse(count=0),  # no tasks created in last day
            TaskCountResponse(count=0),  # no done tasks
        ]

        result = await auto_scaler.get_desired_workers(num_workers=2)

        # With 0 tasks per day, should need 0 workers (but will be capped at minimum by orchestrator)
        expected = math.ceil(0 * 120.0 / (60 * 60 * 24) * 1.2)
        assert result == expected
        assert result == 0
