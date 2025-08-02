"""Task representation for the new curriculum system."""

import logging

logger = logging.getLogger(__name__)


class Task:
    """
    Represents a task selected from the curriculum pool.

    A task has:
    - task_id: unique identifier used to seed task generation
    - slot_id: position in the shared memory pool
    - client: reference to the curriculum client for completion
    """

    def __init__(self, task_id: int, slot_id: int, client: "CurriculumClient"):
        """
        Initialize a task.

        Args:
            task_id: Unique task identifier
            slot_id: Position in the shared memory pool
            client: CurriculumClient instance for task completion
        """
        self._task_id = task_id
        self._slot_id = slot_id
        self._client = client
        self._is_complete = False

        logger.debug(f"Created task {task_id} at slot {slot_id}")

    @property
    def task_id(self) -> int:
        """Get the task ID."""
        return self._task_id

    def complete(self, reward_mean: float, reward_var: float):
        """
        Complete the task and update its score in the curriculum.

        Args:
            reward_mean: Mean reward achieved on this task
            reward_var: Variance of rewards achieved on this task
        """
        if self._is_complete:
            logger.warning(f"Task {self._task_id} is already complete")
            return

        self._client.complete_task(self._slot_id, self._task_id, reward_mean, reward_var)
        self._is_complete = True

        logger.debug(f"Completed task {self._task_id}: reward_mean={reward_mean:.3f}, reward_var={reward_var:.3f}")

    def is_complete(self) -> bool:
        """Check if the task has been completed."""
        return self._is_complete

    def __repr__(self) -> str:
        """String representation of the task."""
        status = "complete" if self._is_complete else "pending"
        return f"Task(id={self._task_id}, slot={self._slot_id}, status={status})"
