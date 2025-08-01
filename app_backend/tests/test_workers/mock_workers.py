import asyncio
import logging
from typing import Any

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.eval_task_worker import AbstractTaskExecutor, EvalTaskWorker
from metta.app_backend.routes.eval_task_routes import TaskResponse


class MockTaskExecutor(AbstractTaskExecutor):
    """Base mock task executor for testing."""
    
    def __init__(
        self,
        sim_delay: float = 0.1,
        logger: logging.Logger | None = None,
    ):
        self.sim_delay = sim_delay
        self._logger = logger or logging.getLogger(__name__)
        self.processed_tasks: list[TaskResponse] = []

    async def execute_task(self, task: TaskResponse) -> None:
        """Base implementation - should be overridden by subclasses."""
        self._logger.info(f"MockTaskExecutor processing task {task.id}")
        await asyncio.sleep(self.sim_delay)
        self.processed_tasks.append(task)
        self._logger.info(f"MockTaskExecutor completed task {task.id} successfully")


class MockSuccessTaskExecutor(MockTaskExecutor):
    """Mock task executor that always succeeds."""
    
    async def execute_task(self, task: TaskResponse) -> None:
        """Override to simulate successful task completion."""
        self._logger.info(f"MockSuccessTaskExecutor processing task {task.id}")
        
        # Simulate some work
        await asyncio.sleep(self.sim_delay)
        
        # Track processed tasks
        self.processed_tasks.append(task)
        
        self._logger.info(f"MockSuccessTaskExecutor completed task {task.id} successfully")


class MockFailureTaskExecutor(MockTaskExecutor):
    """Mock task executor that always fails."""
    
    def __init__(
        self,
        failure_message: str = "Simulated failure",
        sim_delay: float = 0.1,
        logger: logging.Logger | None = None,
    ):
        super().__init__(sim_delay, logger)
        self.failure_message = failure_message
        self.failed_tasks: list[TaskResponse] = []

    async def execute_task(self, task: TaskResponse) -> None:
        """Override to simulate task failure."""
        self._logger.info(f"MockFailureTaskExecutor processing task {task.id}")
        
        # Simulate some work before failing
        await asyncio.sleep(self.sim_delay)
        
        # Track failed tasks
        self.failed_tasks.append(task)
        
        # Always raise an exception
        raise RuntimeError(self.failure_message)


class MockTimeoutTaskExecutor(MockTaskExecutor):
    """Mock task executor that simulates long-running tasks (for timeout testing)."""
    
    def __init__(
        self,
        sim_delay: float = 30.0,  # Long delay to simulate timeout
        logger: logging.Logger | None = None,
    ):
        super().__init__(sim_delay, logger)
        self.started_tasks: list[TaskResponse] = []

    async def execute_task(self, task: TaskResponse) -> None:
        """Override to simulate long-running task."""
        self._logger.info(f"MockTimeoutTaskExecutor processing task {task.id} (will run for {self.sim_delay}s)")
        
        # Track started tasks
        self.started_tasks.append(task)
        
        # Simulate very long work that should timeout
        await asyncio.sleep(self.sim_delay)
        
        self._logger.info(f"MockTimeoutTaskExecutor completed task {task.id} (this shouldn't happen in timeout tests)")


class MockConditionalTaskExecutor(MockTaskExecutor):
    """Mock task executor with configurable success/failure behavior."""
    
    def __init__(
        self,
        success_condition: Any = lambda task: True,  # Function that determines success/failure
        failure_message: str = "Conditional failure",
        sim_delay: float = 0.1,
        logger: logging.Logger | None = None,
    ):
        super().__init__(sim_delay, logger)
        self.success_condition = success_condition
        self.failure_message = failure_message
        self.failed_tasks: list[TaskResponse] = []

    async def execute_task(self, task: TaskResponse) -> None:
        """Override to simulate conditional success/failure."""
        self._logger.info(f"MockConditionalTaskExecutor processing task {task.id}")
        
        # Simulate some work
        await asyncio.sleep(self.sim_delay)
        
        # Check condition
        if self.success_condition(task):
            self.processed_tasks.append(task)
            self._logger.info(f"MockConditionalTaskExecutor completed task {task.id} successfully")
        else:
            self.failed_tasks.append(task)
            raise RuntimeError(f"{self.failure_message} for task {task.id}")


# Convenience worker classes for backward compatibility
class MockSuccessWorker(EvalTaskWorker):
    """Mock worker that always succeeds in task execution."""
    
    def __init__(
        self,
        client: EvalTaskClient,
        assignee: str,
        sim_delay: float = 0.1,
        logger: logging.Logger | None = None,
    ):
        task_executor = MockSuccessTaskExecutor(sim_delay=sim_delay, logger=logger)
        super().__init__(client, task_executor, assignee, logger)
        # Expose executor for test access
        self.task_executor = task_executor

    @property 
    def processed_tasks(self) -> list[TaskResponse]:
        """Access to processed tasks for testing."""
        return self.task_executor.processed_tasks


class MockFailureWorker(EvalTaskWorker):
    """Mock worker that always fails in task execution."""
    
    def __init__(
        self,
        client: EvalTaskClient,
        assignee: str,
        failure_message: str = "Simulated failure",
        sim_delay: float = 0.1,
        logger: logging.Logger | None = None,
    ):
        task_executor = MockFailureTaskExecutor(
            failure_message=failure_message,
            sim_delay=sim_delay,
            logger=logger
        )
        super().__init__(client, task_executor, assignee, logger)
        # Expose executor for test access
        self.task_executor = task_executor

    @property
    def failed_tasks(self) -> list[TaskResponse]:
        """Access to failed tasks for testing."""
        return self.task_executor.failed_tasks


class MockTimeoutWorker(EvalTaskWorker):
    """Mock worker that simulates long-running tasks (for timeout testing)."""
    
    def __init__(
        self,
        client: EvalTaskClient,
        assignee: str,
        sim_delay: float = 30.0,
        logger: logging.Logger | None = None,
    ):
        task_executor = MockTimeoutTaskExecutor(sim_delay=sim_delay, logger=logger)
        super().__init__(client, task_executor, assignee, logger)
        # Expose executor for test access
        self.task_executor = task_executor

    @property
    def started_tasks(self) -> list[TaskResponse]:
        """Access to started tasks for testing."""
        return self.task_executor.started_tasks


class MockConditionalWorker(EvalTaskWorker):
    """Mock worker with configurable success/failure behavior."""
    
    def __init__(
        self,
        client: EvalTaskClient,
        assignee: str,
        success_condition: Any = lambda task: True,
        failure_message: str = "Conditional failure",
        sim_delay: float = 0.1,
        logger: logging.Logger | None = None,
    ):
        task_executor = MockConditionalTaskExecutor(
            success_condition=success_condition,
            failure_message=failure_message,
            sim_delay=sim_delay,
            logger=logger
        )
        super().__init__(client, task_executor, assignee, logger)
        # Expose executor for test access
        self.task_executor = task_executor

    @property
    def processed_tasks(self) -> list[TaskResponse]:
        """Access to processed tasks for testing."""
        return self.task_executor.processed_tasks

    @property
    def failed_tasks(self) -> list[TaskResponse]:
        """Access to failed tasks for testing."""
        return self.task_executor.failed_tasks


class MockWorkerFactory:
    """Factory for creating different types of mock workers."""
    
    @staticmethod
    def create_success_worker(**kwargs):
        """Create a MockSuccessWorker factory with given parameters."""
        def factory(client: EvalTaskClient, assignee: str):
            return MockSuccessWorker(client, assignee, **kwargs)
        return factory
    
    @staticmethod
    def create_failure_worker(**kwargs):
        """Create a MockFailureWorker factory with given parameters."""
        def factory(client: EvalTaskClient, assignee: str):
            return MockFailureWorker(client, assignee, **kwargs)
        return factory
    
    @staticmethod
    def create_timeout_worker(**kwargs):
        """Create a MockTimeoutWorker factory with given parameters."""
        def factory(client: EvalTaskClient, assignee: str):
            return MockTimeoutWorker(client, assignee, **kwargs)
        return factory
    
    @staticmethod
    def create_conditional_worker(**kwargs):
        """Create a MockConditionalWorker factory with given parameters."""
        def factory(client: EvalTaskClient, assignee: str):
            return MockConditionalWorker(client, assignee, **kwargs)
        return factory