import asyncio
import logging
from typing import Any

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.eval_task_worker import EvalTaskWorker
from metta.app_backend.routes.eval_task_routes import TaskResponse


class MockSuccessWorker(EvalTaskWorker):
    """Mock worker that always succeeds in _run_sim_task()."""
    
    def __init__(
        self, 
        client: EvalTaskClient, 
        assignee: str, 
        backend_url: str,
        machine_token: str | None = None, 
        logger: logging.Logger | None = None,
        sim_delay: float = 0.1  # Short delay to simulate work
    ):
        super().__init__(client, assignee, backend_url, machine_token, logger)
        self.sim_delay = sim_delay
        self.processed_tasks: list[TaskResponse] = []

    async def _run_sim_task(
        self,
        task: TaskResponse,
        sim_suite: str,
        env_overrides: dict,
    ) -> None:
        """Override to simulate successful task completion."""
        self._logger.info(f"MockSuccessWorker processing task {task.id} with sim_suite={sim_suite}")
        
        # Simulate some work
        await asyncio.sleep(self.sim_delay)
        
        # Track processed tasks
        self.processed_tasks.append(task)
        
        self._logger.info(f"MockSuccessWorker completed task {task.id} successfully")


class MockFailureWorker(EvalTaskWorker):
    """Mock worker that always fails in _run_sim_task()."""
    
    def __init__(
        self, 
        client: EvalTaskClient, 
        assignee: str, 
        backend_url: str,
        machine_token: str | None = None, 
        logger: logging.Logger | None = None,
        failure_message: str = "Simulated failure",
        sim_delay: float = 0.1
    ):
        super().__init__(client, assignee, backend_url, machine_token, logger)
        self.failure_message = failure_message
        self.sim_delay = sim_delay
        self.failed_tasks: list[TaskResponse] = []

    async def _run_sim_task(
        self,
        task: TaskResponse,
        sim_suite: str,
        env_overrides: dict,
    ) -> None:
        """Override to simulate task failure."""
        self._logger.info(f"MockFailureWorker processing task {task.id}")
        
        # Simulate some work before failing
        await asyncio.sleep(self.sim_delay)
        
        # Track failed tasks
        self.failed_tasks.append(task)
        
        # Always raise an exception
        raise RuntimeError(self.failure_message)


class MockTimeoutWorker(EvalTaskWorker):
    """Mock worker that simulates long-running tasks (for timeout testing)."""
    
    def __init__(
        self, 
        client: EvalTaskClient, 
        assignee: str, 
        backend_url: str,
        machine_token: str | None = None, 
        logger: logging.Logger | None = None,
        sim_delay: float = 30.0  # Long delay to simulate timeout
    ):
        super().__init__(client, assignee, backend_url, machine_token, logger)
        self.sim_delay = sim_delay
        self.started_tasks: list[TaskResponse] = []

    async def _run_sim_task(
        self,
        task: TaskResponse,
        sim_suite: str,
        env_overrides: dict,
    ) -> None:
        """Override to simulate long-running task."""
        self._logger.info(f"MockTimeoutWorker processing task {task.id} (will run for {self.sim_delay}s)")
        
        # Track started tasks
        self.started_tasks.append(task)
        
        # Simulate very long work that should timeout
        await asyncio.sleep(self.sim_delay)
        
        self._logger.info(f"MockTimeoutWorker completed task {task.id} (this shouldn't happen in timeout tests)")


class MockConditionalWorker(EvalTaskWorker):
    """Mock worker with configurable success/failure behavior."""
    
    def __init__(
        self, 
        client: EvalTaskClient, 
        assignee: str, 
        backend_url: str,
        machine_token: str | None = None, 
        logger: logging.Logger | None = None,
        success_condition: Any = lambda task: True,  # Function that determines success/failure
        failure_message: str = "Conditional failure",
        sim_delay: float = 0.1
    ):
        super().__init__(client, assignee, backend_url, machine_token, logger)
        self.success_condition = success_condition
        self.failure_message = failure_message
        self.sim_delay = sim_delay
        self.processed_tasks: list[TaskResponse] = []
        self.failed_tasks: list[TaskResponse] = []

    async def _run_sim_task(
        self,
        task: TaskResponse,
        sim_suite: str,
        env_overrides: dict,
    ) -> None:
        """Override to simulate conditional success/failure."""
        self._logger.info(f"MockConditionalWorker processing task {task.id}")
        
        # Simulate some work
        await asyncio.sleep(self.sim_delay)
        
        # Check condition
        if self.success_condition(task):
            self.processed_tasks.append(task)
            self._logger.info(f"MockConditionalWorker completed task {task.id} successfully")
        else:
            self.failed_tasks.append(task)
            raise RuntimeError(f"{self.failure_message} for task {task.id}")


class MockWorkerFactory:
    """Factory for creating different types of mock workers."""
    
    @staticmethod
    def create_success_worker(**kwargs) -> type[MockSuccessWorker]:
        """Create a MockSuccessWorker class with given parameters."""
        def factory(client: EvalTaskClient, assignee: str, backend_url: str, machine_token: str | None = None, logger: logging.Logger | None = None):
            return MockSuccessWorker(client, assignee, backend_url, machine_token, logger, **kwargs)
        return factory
    
    @staticmethod
    def create_failure_worker(**kwargs) -> type[MockFailureWorker]:
        """Create a MockFailureWorker class with given parameters."""
        def factory(client: EvalTaskClient, assignee: str, backend_url: str, machine_token: str | None = None, logger: logging.Logger | None = None):
            return MockFailureWorker(client, assignee, backend_url, machine_token, logger, **kwargs)
        return factory
    
    @staticmethod
    def create_timeout_worker(**kwargs) -> type[MockTimeoutWorker]:
        """Create a MockTimeoutWorker class with given parameters."""
        def factory(client: EvalTaskClient, assignee: str, backend_url: str, machine_token: str | None = None, logger: logging.Logger | None = None):
            return MockTimeoutWorker(client, assignee, backend_url, machine_token, logger, **kwargs)
        return factory
    
    @staticmethod
    def create_conditional_worker(**kwargs) -> type[MockConditionalWorker]:
        """Create a MockConditionalWorker class with given parameters."""
        def factory(client: EvalTaskClient, assignee: str, backend_url: str, machine_token: str | None = None, logger: logging.Logger | None = None):
            return MockConditionalWorker(client, assignee, backend_url, machine_token, logger, **kwargs)
        return factory