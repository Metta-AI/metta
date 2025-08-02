"""Curriculum Client for workers - selects tasks from shared memory pool."""

import logging
from multiprocessing import shared_memory
from typing import List, Tuple

import numpy as np

from .manager import CurriculumManager, TaskState
from .task import Task

logger = logging.getLogger(__name__)


class CurriculumClient:
    """
    Worker-side curriculum client that selects tasks from the shared memory pool.

    The client:
    1. Replaces the task with the lowest score when it has enough runs
    2. Samples tasks from the pool and selects one based on the strategy
    3. Returns a Task object that can complete itself

    Connects to the curriculum manager by name (default: "metta/curriculum/tasks").
    Special characters in the name (/, .) are automatically converted to underscores.
    """

    def __init__(
        self,
        curriculum_name: str,
        pool_size: int,
        num_samples: int = 16,  # how many tasks to consider for selection
        min_runs: int = 10,
        selection_strategy: str = "epsilon_greedy",
        epsilon: float = 0.1,
        temperature: float = 1.0,
        ucb_c: float = 2.0,  # exploration constant for UCB
    ):
        """
        Initialize the curriculum client.

        Args:
            curriculum_name: Name of the shared memory to connect to
            pool_size: Size of the task pool
            num_samples: Number of tasks to sample for selection
            min_runs: Minimum runs before a task can be replaced
            selection_strategy: Strategy for task selection ("epsilon_greedy", "softmax", "ucb")
            epsilon: Exploration rate for epsilon-greedy
            temperature: Temperature for softmax selection
            ucb_c: Exploration constant for UCB selection
        """
        self.curriculum_name = curriculum_name
        self.pool_size = pool_size
        self.num_samples = min(num_samples, pool_size)
        self.min_runs = min_runs
        self.selection_strategy = selection_strategy
        self.epsilon = epsilon
        self.temperature = temperature
        self.ucb_c = ucb_c

        # Connect to shared memory
        # Ensure name is valid for shared memory (no special chars that might be interpreted as paths)
        safe_name = curriculum_name.replace("/", "_").replace(".", "_")
        self._shm = shared_memory.SharedMemory(name=f"{safe_name}_pool")
        self.pool = np.ndarray((pool_size * CurriculumManager.FIELDS_PER_TASK,), dtype=np.float64, buffer=self._shm.buf)

        # Connect to stats shared memory
        self._stats_shm = shared_memory.SharedMemory(name=f"{safe_name}_stats")
        self._stats = np.ndarray((2,), dtype=np.int64, buffer=self._stats_shm.buf)

        # Track total selections for UCB
        self._total_selections = 0

        logger.info(f"CurriculumClient initialized with strategy={selection_strategy}, num_samples={num_samples}")

    def _get_task_state(self, slot_id: int) -> TaskState:
        """Get TaskState from shared memory."""
        base_idx = slot_id * CurriculumManager.FIELDS_PER_TASK
        return TaskState(
            task_id=int(self.pool[base_idx + CurriculumManager.TASK_ID_IDX]),
            score=self.pool[base_idx + CurriculumManager.SCORE_IDX],
            num_runs=int(self.pool[base_idx + CurriculumManager.NUM_RUNS_IDX]),
            last_update=self.pool[base_idx + CurriculumManager.LAST_UPDATE_IDX],
            reward_mean=self.pool[base_idx + CurriculumManager.REWARD_MEAN_IDX],
            reward_var=self.pool[base_idx + CurriculumManager.REWARD_VAR_IDX],
        )

    def _set_task_state(self, slot_id: int, task_state: TaskState):
        """Set TaskState in shared memory."""
        base_idx = slot_id * CurriculumManager.FIELDS_PER_TASK
        self.pool[base_idx + CurriculumManager.TASK_ID_IDX] = float(task_state.task_id)
        self.pool[base_idx + CurriculumManager.SCORE_IDX] = task_state.score
        self.pool[base_idx + CurriculumManager.NUM_RUNS_IDX] = float(task_state.num_runs)
        self.pool[base_idx + CurriculumManager.LAST_UPDATE_IDX] = task_state.last_update
        self.pool[base_idx + CurriculumManager.REWARD_MEAN_IDX] = task_state.reward_mean
        self.pool[base_idx + CurriculumManager.REWARD_VAR_IDX] = task_state.reward_var

    def get_task(self) -> Task:
        """Sample tasks and return the best one based on strategy."""
        # First, check if we need to replace the lowest scoring task
        all_task_states = [(i, self._get_task_state(i)) for i in range(self.pool_size)]

        # Find task with lowest score that has min_runs
        eligible_for_removal = [(i, ts) for i, ts in all_task_states if ts.num_runs >= self.min_runs]

        if eligible_for_removal:
            # Sort by score (ascending - lowest first)
            eligible_for_removal.sort(key=lambda x: x[1].score)
            lowest_score_idx, lowest_score_task = eligible_for_removal[0]

            # Replace the lowest scoring task with a new one
            new_task_id = np.random.randint(0, 2**31 - 1)
            new_task_state = TaskState(task_id=new_task_id, last_update=self._get_current_time())
            self._set_task_state(lowest_score_idx, new_task_state)

            logger.debug(
                f"Replaced task {lowest_score_task.task_id} (score={lowest_score_task.score:.3f}) "
                f"with new task {new_task_id}"
            )

        # Now sample tasks for selection
        sample_indices = np.random.choice(self.pool_size, self.num_samples, replace=False)
        sampled_tasks = [(i, self._get_task_state(i)) for i in sample_indices]

        # Select task based on strategy
        selected_idx = self._select_task(sampled_tasks)
        slot_id, selected_task = sampled_tasks[selected_idx]

        self._total_selections += 1

        return Task(task_id=selected_task.task_id, slot_id=slot_id, client=self)

    def _select_task(self, sampled_tasks: List[Tuple[int, TaskState]]) -> int:
        """Select a task from the sampled tasks based on the strategy."""
        if self.selection_strategy == "epsilon_greedy":
            return self._epsilon_greedy_selection(sampled_tasks)
        elif self.selection_strategy == "softmax":
            return self._softmax_selection(sampled_tasks)
        elif self.selection_strategy == "ucb":
            return self._ucb_selection(sampled_tasks)
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    def _softmax_selection(self, sampled_tasks: List[Tuple[int, TaskState]]) -> int:
        """Softmax selection: probabilistic selection based on scores."""
        # Use inverse scores (lower score = higher probability)
        scores = np.array([1.0 / (ts.score + 1e-6) for _, ts in sampled_tasks])

        # Apply temperature
        scores = scores / self.temperature

        # Compute softmax probabilities
        exp_scores = np.exp(scores - np.max(scores))  # subtract max for numerical stability
        probs = exp_scores / np.sum(exp_scores)

        return np.random.choice(len(sampled_tasks), p=probs)

    def _epsilon_greedy_selection(self, sampled_tasks: List[Tuple[int, TaskState]]) -> int:
        """Epsilon-greedy selection: exploit best or explore randomly."""
        if np.random.random() < self.epsilon:
            # Explore: random selection
            return np.random.randint(len(sampled_tasks))
        else:
            # Exploit: select task with lowest score
            scores = [ts.score for _, ts in sampled_tasks]
            return int(np.argmin(scores))

    def _ucb_selection(self, sampled_tasks: List[Tuple[int, TaskState]]) -> int:
        """Upper Confidence Bound selection: balance exploitation and exploration."""
        ucb_scores = []

        for _, task in sampled_tasks:
            if task.num_runs == 0:
                # Unplayed tasks have infinite UCB score
                ucb_scores.append(float("inf"))
            else:
                # UCB = mean_reward + c * sqrt(ln(total) / n)
                # We use negative score as a proxy for reward
                exploit_term = -task.score
                explore_term = self.ucb_c * np.sqrt(np.log(self._total_selections + 1) / task.num_runs)
                ucb_scores.append(exploit_term + explore_term)

        # Select task with highest UCB score
        return int(np.argmax(ucb_scores))

    def complete_task(self, slot_id: int, task_id: int, reward_mean: float, reward_var: float):
        """Complete a task and update its score."""
        base_idx = slot_id * CurriculumManager.FIELDS_PER_TASK

        # Verify this is still the same task
        stored_task_id = int(self.pool[base_idx + CurriculumManager.TASK_ID_IDX])
        if stored_task_id != task_id:
            logger.warning(f"Task mismatch: expected {task_id}, found {stored_task_id}")
            return  # Task was already replaced

        # Get current num_runs for score calculation
        current_num_runs = int(self.pool[base_idx + CurriculumManager.NUM_RUNS_IDX])

        # Update task statistics directly in shared memory (lockless)
        new_num_runs = current_num_runs + 1
        score = self._compute_score(reward_mean, reward_var, new_num_runs)

        # Direct updates to shared memory
        self.pool[base_idx + CurriculumManager.SCORE_IDX] = score
        self.pool[base_idx + CurriculumManager.NUM_RUNS_IDX] = float(new_num_runs)
        self.pool[base_idx + CurriculumManager.REWARD_MEAN_IDX] = reward_mean
        self.pool[base_idx + CurriculumManager.REWARD_VAR_IDX] = reward_var
        self.pool[base_idx + CurriculumManager.LAST_UPDATE_IDX] = self._get_current_time()

        logger.debug(f"Completed task {task_id}: score={score:.3f}, reward_mean={reward_mean:.3f}, runs={new_num_runs}")

        # Update statistics (atomic increment)
        self._stats[0] += 1  # increment total_completions

    def _compute_score(self, reward_mean: float, reward_var: float, num_runs: int) -> float:
        """
        Compute task score - lower is easier/less interesting.

        The score combines:
        - Task difficulty (1 - reward_mean): easier tasks have lower scores
        - Uncertainty (normalized variance): high variance indicates unstable performance
        - Confidence: increases with more samples
        """
        # Confidence increases with more samples
        confidence = 1.0 - np.exp(-num_runs / 10.0)

        # Normalize variance by mean to make it scale-invariant
        normalized_var = reward_var / (abs(reward_mean) + 1e-6)

        # Score combines difficulty and uncertainty
        # Lower score = easier task = less interesting for curriculum
        difficulty_score = 1.0 - reward_mean  # 0 = very hard, 1 = very easy
        uncertainty_penalty = normalized_var * 0.2  # add some weight to variance

        score = difficulty_score * confidence + uncertainty_penalty * (1.0 - confidence)

        return np.clip(score, 0.0, 1.0)

    def _get_current_time(self) -> float:
        """Get current time for consistency."""
        import time

        return time.time()

    def get_stats(self) -> dict:
        """Get client statistics."""
        return {
            "selection_strategy": self.selection_strategy,
            "num_samples": self.num_samples,
            "total_selections": self._total_selections,
            "epsilon": self.epsilon if self.selection_strategy == "epsilon_greedy" else None,
            "temperature": self.temperature if self.selection_strategy == "softmax" else None,
            "ucb_c": self.ucb_c if self.selection_strategy == "ucb" else None,
        }

    def cleanup(self):
        """Clean up shared memory connections."""
        try:
            self._shm.close()
            self._stats_shm.close()
            logger.debug(f"Cleaned up shared memory connections for curriculum {self.curriculum_name}")
        except Exception as e:
            logger.warning(f"Error cleaning up shared memory: {e}")
