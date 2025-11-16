"""Task performance tracking with pluggable memory backends and dual-pool support.

This module provides task tracking infrastructure for curriculum learning, supporting both
single-pool and dual-pool architectures. The implementation is unified - the same code works
whether using local memory (single-process) or shared memory (multi-process).

Core Classes:
    - TaskTracker: Single-pool task tracking with configurable memory backend
    - DualPoolTaskTracker: Dual-pool architecture with separate explore/exploit pools

Key Responsibilities:
    - Track task creation and removal with O(1) lookup by task_id
    - Maintain exponential moving averages of task performance
    - Provide thread-safe access via backend-specific locking
    - Support state serialization for checkpointing
    - Enable atomic task promotion between pools (dual-pool mode)

Memory Backend Abstraction:
    - LocalMemoryBackend: Fast numpy arrays for single-process training
    - SharedMemoryBackend: Multiprocessing shared memory for distributed workers
    - Backend selection is transparent to curriculum algorithms

Dual-Pool Architecture:
    The DualPoolTaskTracker manages two independent TaskTracker instances:
    - Explore pool: Smaller, high-turnover pool for discovering new learning opportunities
    - Exploit pool: Larger, selective pool for tasks with proven learning progress

    Features:
    - Atomic task promotion with data preservation (all 18 float64 values)
    - Independent shared memory regions per pool
    - Pool-aware task routing and statistics
    - Full state persistence for checkpointing

Why Separate File:
    Task tracking is a distinct concern from curriculum logic. It manages the low-level
    storage and updates, while curriculum algorithms make high-level decisions about
    what to track and how to use the tracked data.

See Also:
    - learning_progress_algorithm.py: Uses TaskTracker/DualPoolTaskTracker for performance data
    - shared_memory_backend.py: Memory backend implementations
"""

import time
from typing import Any, Dict, List, Optional

from metta.cogworks.curriculum.shared_memory_backend import LocalMemoryBackend, SharedMemoryBackend, TaskMemoryBackend


class TaskTracker:
    """Unified task tracker using configurable memory backend.

    Works with both local (single-process) and shared (multi-process) memory backends.
    All task tracking logic is unified - no conditional branching based on backend type.
    """

    def __init__(
        self,
        max_memory_tasks: int = 1000,
        ema_alpha: float = 0.1,
        backend: Optional[TaskMemoryBackend] = None,
        session_id: Optional[str] = None,
        use_shared_memory: bool = False,
        task_struct_size: int = 18,
        default_success_threshold: float = 0.5,
        default_generator_type: float = 0.0,
    ):
        """Initialize task tracker with configurable backend.

        Args:
            max_memory_tasks: Maximum number of tasks to track
            ema_alpha: Alpha parameter for exponential moving average
            backend: Optional pre-configured backend. If None, creates based on use_shared_memory
            session_id: Unique identifier for shared memory session (only for shared memory)
            use_shared_memory: If True and backend is None, creates SharedMemoryBackend
            task_struct_size: Size of each task's data structure (default: 18)
            default_success_threshold: Default success threshold for new tasks (default: 0.5)
            default_generator_type: Default generator type identifier (default: 0.0)
        """
        self.max_memory_tasks = max_memory_tasks
        self.ema_alpha = ema_alpha
        self.default_success_threshold = default_success_threshold
        self.default_generator_type = default_generator_type

        # Running statistics for global tracking (replaces completion_history buffer)
        self._total_completions = 0
        self._sum_scores = 0.0

        # Initialize or use provided backend
        if backend is None:
            if use_shared_memory:
                backend = SharedMemoryBackend(
                    max_tasks=max_memory_tasks,
                    session_id=session_id,
                    task_struct_size=task_struct_size,
                )
            else:
                backend = LocalMemoryBackend(
                    max_tasks=max_memory_tasks,
                    task_struct_size=task_struct_size,
                )

        self._backend: TaskMemoryBackend = backend
        self._task_id_to_index: Dict[int, int] = {}
        self._next_free_index = 0

        # Label tracking: hash -> label string mapping (local, not in shared memory)
        self._label_hash_to_string: Dict[int, str] = {}

        # Rebuild mapping from existing memory
        self._rebuild_task_mapping()

    def _rebuild_task_mapping(self) -> None:
        """Rebuild task ID to array index mapping by scanning backend memory."""
        with self._backend.acquire_lock():
            self._task_id_to_index.clear()
            self._next_free_index = self._backend.max_tasks  # Default to end
            first_free_index = None

            for i in range(self._backend.max_tasks):
                task_data = self._backend.get_task_data(i)
                task_id = int(task_data[0])
                is_active = bool(task_data[12])

                if is_active and task_id > 0:
                    self._task_id_to_index[task_id] = i
                elif task_id == 0 and first_free_index is None:
                    # Track first free slot but keep scanning for active tasks
                    first_free_index = i

            # Use first free slot if found, otherwise pool is full
            if first_free_index is not None:
                self._next_free_index = first_free_index

    def track_task_creation(
        self,
        task_id: int,
        success_threshold: Optional[float] = None,
        seed: Optional[float] = None,
        generator_type: Optional[float] = None,
    ) -> None:
        """Track when a task is created with metadata."""
        with self._backend.acquire_lock():
            timestamp = time.time()
            if seed is None:
                seed = hash(str(task_id) + str(timestamp)) % (2**31)
            if success_threshold is None:
                success_threshold = self.default_success_threshold
            if generator_type is None:
                generator_type = self.default_generator_type

            # Check if task already exists
            if task_id in self._task_id_to_index:
                return

            # Find slot in backend memory
            if self._next_free_index >= self._backend.max_tasks:
                return  # No space available

            index = self._next_free_index
            self._task_id_to_index[task_id] = index

            # Write to backend memory
            task_data = self._backend.get_task_data(index)
            task_data[0] = float(task_id)
            task_data[1] = timestamp
            task_data[2] = 0.0  # completion_count
            task_data[3] = 0.0  # reward_ema
            task_data[4] = 0.0  # lp_score
            task_data[5] = 0.0  # success_rate_ema
            task_data[6] = 0.0  # total_score
            task_data[7] = 0.0  # last_score
            task_data[8] = success_threshold
            task_data[9] = float(seed)
            task_data[10] = generator_type
            task_data[11] = 0.0  # ema_squared (for variance calculation)
            task_data[12] = 1.0  # is_active
            # Bidirectional LP EMAs (indices 13-16)
            task_data[13] = 0.0  # p_fast
            task_data[14] = 0.0  # p_slow
            task_data[15] = 0.0  # p_true
            task_data[16] = 0.0  # random_baseline

            # Find next free slot after this one
            self._next_free_index = index + 1
            while self._next_free_index < self._backend.max_tasks:
                next_task_data = self._backend.get_task_data(self._next_free_index)
                if next_task_data[0] == 0.0:  # Slot is free (task_id == 0)
                    break
                self._next_free_index += 1

    def update_task_performance(
        self,
        task_id: int,
        score: float,
        lp_score: Optional[float] = None,
        success_threshold: Optional[float] = None,
    ) -> None:
        """Update task performance with new completion score.

        NOTE: This method is kept for backward compatibility. New code should use
        update_task_performance_with_bidirectional_emas() for atomic updates.
        """
        # Create task if needed (outside the main lock to avoid deadlock)
        if task_id not in self._task_id_to_index:
            self.track_task_creation(task_id, success_threshold=success_threshold or 0.5)

        with self._backend.acquire_lock():
            # Task should exist now
            if task_id not in self._task_id_to_index:
                # Race condition - another process might have removed it
                return

            index = self._task_id_to_index[task_id]
            task_data = self._backend.get_task_data(index)

            # Read current values
            completion_count = int(task_data[2])
            reward_ema = task_data[3]
            old_lp_score = task_data[4]
            success_rate_ema = task_data[5]
            total_score = task_data[6]
            task_success_threshold = task_data[8]
            ema_squared = task_data[11]

            # Update counts and totals
            new_completion_count = completion_count + 1
            new_total_score = total_score + score

            # Update reward EMA
            if completion_count == 0:
                new_reward_ema = score
            else:
                new_reward_ema = (1 - self.ema_alpha) * reward_ema + self.ema_alpha * score

            # Update EMA of squared scores (for variance calculation)
            score_squared = score * score
            if completion_count == 0:
                new_ema_squared = score_squared
            else:
                new_ema_squared = (1 - self.ema_alpha) * ema_squared + self.ema_alpha * score_squared

            # Update LP score if provided
            new_lp_score = lp_score if lp_score is not None else old_lp_score

            # Update success rate EMA
            current_threshold = success_threshold if success_threshold is not None else task_success_threshold
            is_success = float(score >= current_threshold)
            if completion_count == 0:
                new_success_rate_ema = is_success
            else:
                new_success_rate_ema = (1 - self.ema_alpha) * success_rate_ema + self.ema_alpha * is_success

            # Write updated values
            task_data[2] = float(new_completion_count)
            task_data[3] = new_reward_ema
            task_data[4] = new_lp_score
            task_data[5] = new_success_rate_ema
            task_data[6] = new_total_score
            task_data[7] = score
            task_data[8] = current_threshold
            task_data[11] = new_ema_squared

            # Update running statistics (replaces completion_history)
            self._total_completions += 1
            self._sum_scores += score

    def update_task_performance_with_bidirectional_emas(
        self,
        task_id: int,
        score: float,
        scorer: Optional[Any] = None,
        success_threshold: Optional[float] = None,
    ) -> None:
        """Atomic update: basic EMAs + bidirectional EMAs in ONE lock.

        Stage 3: This consolidates what used to be 2-3 separate lock acquisitions:
        1. Basic EMAs (completion_count, reward_ema, success_rate_ema, ema_squared)
        2. Bidirectional EMAs (p_fast, p_slow, p_true, random_baseline)

        Args:
            task_id: Task to update
            score: Performance score (0.0-1.0)
            scorer: BidirectionalLPScorer instance (optional, for bidirectional EMAs)
            success_threshold: Success threshold for binary classification
        """
        # Create task if needed (outside the main lock to avoid deadlock)
        if task_id not in self._task_id_to_index:
            self.track_task_creation(task_id, success_threshold=success_threshold or 0.5)

        with self._backend.acquire_lock():
            # Task should exist now
            if task_id not in self._task_id_to_index:
                # Race condition - another process might have removed it
                return

            index = self._task_id_to_index[task_id]
            task_data = self._backend.get_task_data(index)

            # === PART 1: Basic EMA updates (same as before) ===
            completion_count = int(task_data[2])
            reward_ema = task_data[3]
            old_lp_score = task_data[4]
            success_rate_ema = task_data[5]
            total_score = task_data[6]
            task_success_threshold = task_data[8]
            ema_squared = task_data[11]

            # Update counts and totals
            new_completion_count = completion_count + 1
            new_total_score = total_score + score

            # Update reward EMA
            if completion_count == 0:
                new_reward_ema = score
            else:
                new_reward_ema = (1 - self.ema_alpha) * reward_ema + self.ema_alpha * score

            # Update EMA of squared scores
            score_squared = score * score
            if completion_count == 0:
                new_ema_squared = score_squared
            else:
                new_ema_squared = (1 - self.ema_alpha) * ema_squared + self.ema_alpha * score_squared

            # Update success rate EMA
            current_threshold = success_threshold if success_threshold is not None else task_success_threshold
            is_success = float(score >= current_threshold)
            if completion_count == 0:
                new_success_rate_ema = is_success
            else:
                new_success_rate_ema = (1 - self.ema_alpha) * success_rate_ema + self.ema_alpha * is_success

            # === PART 2: Bidirectional EMA updates (if scorer provided) ===
            if scorer is not None and hasattr(scorer, "config"):
                # Read current bidirectional EMAs
                p_fast = task_data[13]
                p_slow = task_data[14]
                p_true = task_data[15]
                random_baseline = task_data[16]

                task_success_rate = score

                # Handle baseline normalization if enabled
                if scorer.config.use_baseline_normalization:
                    # Set baseline on first update (capped at 0.75)
                    if random_baseline == 0.0:
                        random_baseline = min(task_success_rate, 0.75)

                    # Calculate normalized "mastery" score
                    improvement_over_baseline = max(task_success_rate - random_baseline, 0.0)
                    total_possible_improvement = max(1.0 - random_baseline, 1e-10)
                    normalized_task_success_rate = improvement_over_baseline / total_possible_improvement
                else:
                    # Use raw success rate
                    normalized_task_success_rate = task_success_rate

                # Initialize or update bidirectional EMAs
                if p_fast == 0.0 and p_slow == 0.0:
                    # First update - initialize to current value
                    p_fast = normalized_task_success_rate
                    p_slow = normalized_task_success_rate
                    p_true = task_success_rate
                else:
                    # Update EMAs
                    p_fast = normalized_task_success_rate * scorer.config.ema_timescale + p_fast * (
                        1.0 - scorer.config.ema_timescale
                    )
                    slow_timescale = scorer.config.ema_timescale * scorer.config.slow_timescale_factor
                    p_slow = normalized_task_success_rate * slow_timescale + p_slow * (1.0 - slow_timescale)
                    p_true = task_success_rate * scorer.config.ema_timescale + p_true * (
                        1.0 - scorer.config.ema_timescale
                    )

                # Write bidirectional EMAs
                task_data[13] = p_fast
                task_data[14] = p_slow
                task_data[15] = p_true
                task_data[16] = random_baseline

                # Update tracked task metrics for wandb (if this is a tracked task)
                if hasattr(scorer, "_tracked_task_ids") and task_id in scorer._tracked_task_ids:
                    if task_id in scorer._tracked_task_metrics:
                        lp = p_fast - p_slow
                        scorer._tracked_task_metrics[task_id].update(
                            {
                                "mean_reward": task_success_rate,
                                "fast_ema": p_fast,
                                "slow_ema": p_slow,
                                "raw_lp": lp,
                                "raw_reward": score,
                                "clamped_reward": score,
                            }
                        )

            # === PART 3: Write all basic values ===
            task_data[2] = float(new_completion_count)
            task_data[3] = new_reward_ema
            task_data[4] = old_lp_score  # LP score updated lazily during sampling
            task_data[5] = new_success_rate_ema
            task_data[6] = new_total_score
            task_data[7] = score
            task_data[8] = current_threshold
            task_data[11] = new_ema_squared

            # Update running statistics
            self._total_completions += 1
            self._sum_scores += score

    def get_task_index(self, task_id: int) -> Optional[int]:
        """Get the array index for a task ID.

        For shared memory backends, will search shared memory if task is not in local mapping.
        Returns None if task is not found.
        """
        # Fast path: check local mapping first
        if task_id in self._task_id_to_index:
            return self._task_id_to_index[task_id]

        # Slow path for shared memory: scan to find task from another worker
        if isinstance(self._backend, SharedMemoryBackend):
            for i in range(self._backend.max_tasks):
                task_data = self._backend.get_task_data(i)
                if int(task_data[0]) == task_id and bool(task_data[12]):  # is_active
                    return i

        return None

    def update_lp_score(self, task_id: int, lp_score: float) -> None:
        """Update the learning progress score for a task."""
        index = self.get_task_index(task_id)
        if index is None:
            return

        with self._backend.acquire_lock():
            task_data = self._backend.get_task_data(index)
            task_data[4] = lp_score

    def get_task_stats(self, task_id: int) -> Optional[Dict[str, float]]:
        """Get statistics for a specific task.

        For shared memory backends, will search shared memory if task is not in local mapping.
        This allows reading stats for tasks created by other workers.

        Note: No locking - may read slightly stale data, but that's acceptable
        for statistics queries to avoid lock contention.
        """
        # Fast path: check local mapping first
        if task_id in self._task_id_to_index:
            index = self._task_id_to_index[task_id]
            task_data = self._backend.get_task_data(index)
        elif isinstance(self._backend, SharedMemoryBackend):
            # Slow path: scan shared memory to find task from another worker
            index = None
            for i in range(self._backend.max_tasks):
                task_data = self._backend.get_task_data(i)
                if int(task_data[0]) == task_id and bool(task_data[12]):  # is_active
                    index = i
                    break

            if index is None:
                return None

            task_data = self._backend.get_task_data(index)
        else:
            # Local memory backend and task not found
            return None

        if task_data[12] == 0:  # not active
            return None

        creation_time = task_data[1]
        completion_count = int(task_data[2])
        reward_ema = task_data[3]
        lp_score = task_data[4]
        success_rate_ema = task_data[5]
        total_score = task_data[6]
        last_score = task_data[7]
        success_threshold = task_data[8]
        seed = task_data[9]
        generator_type = task_data[10]
        ema_squared = task_data[11]
        # Bidirectional LP EMAs (indices 13-16)
        p_fast = task_data[13]
        p_slow = task_data[14]
        p_true = task_data[15]
        random_baseline = task_data[16]

        if completion_count == 0:
            return {
                "completion_count": 0,
                "mean_score": 0.0,
                "reward_ema": 0.0,
                "ema_squared": 0.0,
                "lp_score": 0.0,
                "success_rate_ema": 0.0,
                "last_score": 0.0,
                "success_threshold": success_threshold,
                "seed": seed,
                "generator_type": generator_type,
                "age_seconds": time.time() - creation_time,
                "p_fast": p_fast,
                "p_slow": p_slow,
                "p_true": p_true,
                "random_baseline": random_baseline,
            }

        return {
            "completion_count": completion_count,
            "mean_score": total_score / completion_count,
            "reward_ema": reward_ema,
            "ema_squared": ema_squared,
            "lp_score": lp_score,
            "success_rate_ema": success_rate_ema,
            "last_score": last_score,
            "success_threshold": success_threshold,
            "seed": seed,
            "generator_type": generator_type,
            "age_seconds": time.time() - creation_time,
            "p_fast": p_fast,
            "p_slow": p_slow,
            "p_true": p_true,
            "random_baseline": random_baseline,
        }

    def get_all_tracked_tasks(self) -> List[int]:
        """Get all currently tracked task IDs.

        For shared memory backends, scans shared memory to find ALL tasks from ALL workers.
        For local memory backends, returns tasks from the local mapping.

        Note: No locking - returns snapshot which may be slightly stale.
        """
        # For shared memory, scan to see tasks from all workers
        if isinstance(self._backend, SharedMemoryBackend):
            task_ids = []
            for i in range(self._backend.max_tasks):
                task_data = self._backend.get_task_data(i)
                is_active = bool(task_data[12])
                # A slot is active if is_active flag is True
                # (task_id == 0 with is_active == False means free slot)
                if is_active:
                    task_id = int(task_data[0])
                    task_ids.append(task_id)
            return task_ids
        else:
            # Local memory: use mapping (only this process's tasks)
            return list(self._task_id_to_index.keys())

    def remove_task(self, task_id: int) -> None:
        """Remove a task from tracking."""
        with self._backend.acquire_lock():
            if task_id in self._task_id_to_index:
                index = self._task_id_to_index[task_id]
                task_data = self._backend.get_task_data(index)
                task_data[0] = 0.0  # Clear task_id to mark slot as free
                task_data[12] = 0.0  # is_active = False
                del self._task_id_to_index[task_id]

                # Update _next_free_index to enable slot reuse
                # If we just freed a slot before the current free index, update it
                if index < self._next_free_index:
                    self._next_free_index = index

    def get_global_stats(self) -> Dict[str, float]:
        """Get global performance statistics.

        Note: No locking - statistics may be slightly inconsistent but acceptable
        for monitoring purposes. Avoids lock contention on frequent stat queries.
        """
        if self._total_completions == 0:
            return {
                "mean_score": 0.0,
                "total_completions": 0,
            }

        return {
            "mean_score": self._sum_scores / self._total_completions,
            "total_completions": self._total_completions,
        }

    # ========== Label Tracking (Shared Memory) ==========
    # Labels are stored as hashes in shared memory (index 17) with a local mapping
    # to strings. This keeps shared memory efficient while supporting string labels.

    def set_task_label(self, task_id: int, label: str) -> None:
        """Store task label in shared memory (as hash).

        Args:
            task_id: Task ID to label
            label: Label string (e.g., "lonely_heart", "pack_rat")
        """
        if task_id not in self._task_id_to_index:
            return  # Task doesn't exist

        # Compute stable hash for label
        # Use only 53 bits to ensure exact float64 representation (2^53 - 1)
        # This prevents precision loss when storing as float
        label_hash = hash(label) & 0x1FFFFFFFFFFFFF  # 53-bit mask

        # Store hash in shared memory at index 17
        with self._backend.acquire_lock():
            index = self._task_id_to_index[task_id]
            task_data = self._backend.get_task_data(index)
            task_data[17] = float(label_hash)

        # Maintain local hash-to-string mapping
        self._label_hash_to_string[label_hash] = label

    def get_task_label(self, task_id: int) -> Optional[str]:
        """Get task label from shared memory.

        Args:
            task_id: Task ID to query

        Returns:
            Label string, or None if task not found or label not set
        """
        if task_id not in self._task_id_to_index:
            return None

        index = self._task_id_to_index[task_id]
        task_data = self._backend.get_task_data(index)
        label_hash = int(task_data[17])

        if label_hash == 0:
            return None  # No label set

        return self._label_hash_to_string.get(label_hash)

    def get_label_completion_counts(self) -> Dict[str, int]:
        """Count total completions per label by scanning shared memory.

        Returns:
            Dictionary mapping label -> total completion count
        """
        label_counts: Dict[str, int] = {}

        for _task_id, index in self._task_id_to_index.items():
            task_data = self._backend.get_task_data(index)
            is_active = bool(task_data[12])

            if not is_active:
                continue

            label_hash = int(task_data[17])
            if label_hash == 0:
                continue  # No label

            label = self._label_hash_to_string.get(label_hash)
            if label is None:
                continue  # Hash not in mapping (shouldn't happen)

            completion_count = int(task_data[2])
            label_counts[label] = label_counts.get(label, 0) + completion_count

        return label_counts

    def get_state(self) -> Dict[str, Any]:
        """Get task tracker state for checkpointing.

        Note: No locking - checkpoint may have minor inconsistencies if captured
        during updates, but this is acceptable as checkpoints are infrequent.
        """
        task_memory = {}
        for task_id, index in self._task_id_to_index.items():
            task_data = self._backend.get_task_data(index)
            if task_data[12] > 0:  # is_active
                task_memory[task_id] = {
                    "creation_time": task_data[1],
                    "completion_count": int(task_data[2]),
                    "reward_ema": task_data[3],
                    "lp_score": task_data[4],
                    "success_rate_ema": task_data[5],
                    "total_score": task_data[6],
                    "last_score": task_data[7],
                    "success_threshold": task_data[8],
                    "seed": task_data[9],
                    "generator_type": task_data[10],
                    "ema_squared": task_data[11],
                    "p_fast": task_data[13],
                    "p_slow": task_data[14],
                    "p_true": task_data[15],
                    "random_baseline": task_data[16],
                    "label_hash": task_data[17],
                }

        total_completions = sum(int(self._backend.get_task_data(idx)[2]) for idx in self._task_id_to_index.values())

        # Determine tracker type based on backend
        tracker_type = "centralized" if isinstance(self._backend, SharedMemoryBackend) else "local"
        session_id = getattr(self._backend, "session_id", None)

        return {
            "max_memory_tasks": self.max_memory_tasks,
            "tracker_type": tracker_type,
            "session_id": session_id,
            "task_memory": task_memory,
            "task_creation_order": [],  # Not used with backend approach
            "cached_total_completions": total_completions,
            "cache_valid": True,
            "global_total_completions": self._total_completions,
            "global_sum_scores": self._sum_scores,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load task tracker state from checkpoint."""
        with self._backend.acquire_lock():
            self.max_memory_tasks = state["max_memory_tasks"]

            # Clear backend memory
            self._backend.clear()
            self._task_id_to_index.clear()

            # Restore tasks
            for i, (task_id, task_data) in enumerate(state["task_memory"].items()):
                if i >= self._backend.max_tasks:
                    break

                self._task_id_to_index[int(task_id)] = i
                data = self._backend.get_task_data(i)
                data[0] = float(task_id)
                data[1] = task_data.get("creation_time", time.time())
                data[2] = float(task_data.get("completion_count", 0))
                data[3] = task_data.get("reward_ema", 0.0)
                data[4] = task_data.get("lp_score", 0.0)
                data[5] = task_data.get("success_rate_ema", 0.0)
                data[6] = task_data.get("total_score", 0.0)
                data[7] = task_data.get("last_score", 0.0)
                data[8] = task_data.get("success_threshold", 0.5)
                data[9] = task_data.get("seed", 0.0)
                data[10] = task_data.get("generator_type", 0.0)
                data[11] = task_data.get("ema_squared", 0.0)
                data[12] = 1.0  # is_active
                # Bidirectional LP EMAs (indices 13-16)
                data[13] = task_data.get("p_fast", 0.0)
                data[14] = task_data.get("p_slow", 0.0)
                data[15] = task_data.get("p_true", 0.0)
                data[16] = task_data.get("random_baseline", 0.0)
                # Label hash (index 17)
                data[17] = task_data.get("label_hash", 0.0)

            self._next_free_index = len(state["task_memory"])

            # Restore running statistics
            self._total_completions = state.get("global_total_completions", 0)
            self._sum_scores = state.get("global_sum_scores", 0.0)

    def cleanup_shared_memory(self) -> None:
        """Clean up shared memory resources (only relevant for shared memory backend)."""
        self._backend.cleanup()

    def __del__(self) -> None:
        """Cleanup on destruction."""
        # Only close, don't unlink in destructor
        pass  # Backend handles its own cleanup

    def __getstate__(self):
        """Prepare for pickling - save configuration and mappings."""
        return {
            "max_memory_tasks": self.max_memory_tasks,
            "ema_alpha": self.ema_alpha,
            "default_success_threshold": self.default_success_threshold,
            "default_generator_type": self.default_generator_type,
            "backend": self._backend,  # SharedMemoryBackend has its own pickle support
            "total_completions": self._total_completions,
            "sum_scores": self._sum_scores,
        }

    def __setstate__(self, state):
        """Restore from pickle - reconnect to shared memory and rebuild mappings."""
        self.max_memory_tasks = state["max_memory_tasks"]
        self.ema_alpha = state["ema_alpha"]
        self.default_success_threshold = state["default_success_threshold"]
        self.default_generator_type = state["default_generator_type"]
        self._backend = state["backend"]
        self._total_completions = state["total_completions"]
        self._sum_scores = state["sum_scores"]

        # Initialize label tracking (local per-process)
        self._label_hash_to_string = {}

        # Rebuild task ID to index mapping from shared memory
        self._task_id_to_index = {}
        self._next_free_index = 0
        self._rebuild_task_mapping()


# Backwards compatibility factory functions
def LocalTaskTracker(max_memory_tasks: int = 1000, ema_alpha: float = 0.1) -> TaskTracker:
    """Create a local (single-process) task tracker.

    Factory function for backwards compatibility with existing code.

    Args:
        max_memory_tasks: Maximum number of tasks to track
        ema_alpha: Alpha parameter for exponential moving average

    Returns:
        TaskTracker instance with LocalMemoryBackend
    """
    return TaskTracker(max_memory_tasks=max_memory_tasks, ema_alpha=ema_alpha, use_shared_memory=False)


def CentralizedTaskTracker(
    max_memory_tasks: int = 1000,
    session_id: Optional[str] = None,
    ema_alpha: float = 0.1,
    task_struct_size: int = 18,
) -> TaskTracker:
    """Create a centralized (multi-process) task tracker with shared memory.

    Factory function for backwards compatibility with existing code.

    Args:
        max_memory_tasks: Maximum number of tasks to track
        session_id: Unique identifier for shared memory session
        ema_alpha: Alpha parameter for exponential moving average
        task_struct_size: Size of each task's data structure (default: 18)

    Returns:
        TaskTracker instance with SharedMemoryBackend
    """
    return TaskTracker(
        max_memory_tasks=max_memory_tasks,
        ema_alpha=ema_alpha,
        session_id=session_id,
        use_shared_memory=True,
        task_struct_size=task_struct_size,
    )


class DualPoolTaskTracker:
    """Manages two independent task pools (explore and exploit) for dual-pool curriculum.

    This class wraps two TaskTracker instances with separate shared memory regions:
    - explore_tracker: For exploration tasks (smaller pool, high turnover)
    - exploit_tracker: For exploitation tasks (larger pool, selective)

    Key responsibilities:
    - Route task operations to the correct pool
    - Atomically promote tasks from explore to exploit
    - Track which pool each task belongs to
    - Provide per-pool statistics

    Design: Each pool has its own shared memory region with independent session IDs.
    This allows parallel updates and clean separation of exploration vs exploitation.
    """

    def __init__(
        self,
        num_explore_tasks: int,
        num_exploit_tasks: int,
        ema_alpha: float,
        session_id: str,
        use_shared_memory: bool,
        task_struct_size: int,
        default_success_threshold: float,
        default_generator_type: float,
    ):
        """Initialize dual-pool task tracker.

        Args:
            num_explore_tasks: Capacity of exploration pool
            num_exploit_tasks: Capacity of exploitation pool
            ema_alpha: Alpha parameter for exponential moving average
            session_id: Base session ID (will be suffixed with _explore and _exploit)
            use_shared_memory: Whether to use shared memory backend
            task_struct_size: Size of task data structure (default: 18)
            default_success_threshold: Default success threshold for new tasks
            default_generator_type: Default generator type identifier
        """
        self.num_explore_tasks = num_explore_tasks
        self.num_exploit_tasks = num_exploit_tasks

        # Create separate trackers for explore and exploit pools
        self.explore_tracker = TaskTracker(
            max_memory_tasks=num_explore_tasks,
            ema_alpha=ema_alpha,
            session_id=f"{session_id}_explore" if session_id else None,
            use_shared_memory=use_shared_memory,
            task_struct_size=task_struct_size,
            default_success_threshold=default_success_threshold,
            default_generator_type=default_generator_type,
        )

        self.exploit_tracker = TaskTracker(
            max_memory_tasks=num_exploit_tasks,
            ema_alpha=ema_alpha,
            session_id=f"{session_id}_exploit" if session_id else None,
            use_shared_memory=use_shared_memory,
            task_struct_size=task_struct_size,
            default_success_threshold=default_success_threshold,
            default_generator_type=default_generator_type,
        )

        # Track which pool each task belongs to
        self._task_pool_map: Dict[int, str] = {}  # task_id -> 'explore' or 'exploit'

    def get_pool_tracker(self, task_id: int) -> Optional[TaskTracker]:
        """Get the tracker for the pool containing this task.

        Args:
            task_id: Task ID to look up

        Returns:
            TaskTracker for the pool containing this task, or None if not found
        """
        pool = self._task_pool_map.get(task_id)
        if pool == "explore":
            return self.explore_tracker
        elif pool == "exploit":
            return self.exploit_tracker
        return None

    def track_task_creation(
        self,
        task_id: int,
        pool: str,
        success_threshold: Optional[float] = None,
        seed: Optional[float] = None,
        generator_type: Optional[float] = None,
    ) -> None:
        """Track when a task is created in a specific pool.

        Args:
            task_id: Unique task identifier
            pool: Which pool to create task in ('explore' or 'exploit')
            success_threshold: Success threshold for this task
            seed: Random seed for task generation
            generator_type: Generator type identifier
        """
        if pool == "explore":
            tracker = self.explore_tracker
        elif pool == "exploit":
            tracker = self.exploit_tracker
        else:
            raise ValueError(f"Invalid pool: {pool}. Must be 'explore' or 'exploit'")

        tracker.track_task_creation(task_id, success_threshold, seed, generator_type)
        self._task_pool_map[task_id] = pool

    def promote_task(self, task_id: int) -> bool:
        """Atomically promote a task from explore to exploit pool.

        This performs the following steps atomically:
        1. Read all 18 float64 values from explore pool
        2. Find lowest-scoring task in exploit pool (if full)
        3. Evict lowest-scoring exploit task (if pool is full)
        4. Write promoted task to exploit pool
        5. Remove from explore pool
        6. Update task pool map

        Args:
            task_id: ID of task to promote from explore pool

        Returns:
            True if promotion succeeded, False otherwise

        Raises:
            ValueError: If task is not in explore pool
        """
        # Verify task is in explore pool
        if self._task_pool_map.get(task_id) != "explore":
            raise ValueError(f"Task {task_id} is not in explore pool")

        # Get task data from explore pool
        explore_stats = self.explore_tracker.get_task_stats(task_id)
        if not explore_stats:
            return False

        # Read all 18 float64 values from explore pool
        explore_index = self.explore_tracker.get_task_index(task_id)
        if explore_index is None:
            return False

        # Atomically copy task data (all 18 float64 values)
        with self.explore_tracker._backend.acquire_lock():
            task_data = self.explore_tracker._backend.get_task_data(explore_index).copy()

        # Check if exploit pool is full
        exploit_tasks = self.exploit_tracker.get_all_tracked_tasks()
        if len(exploit_tasks) >= self.num_exploit_tasks:
            # Pool is full - need to evict lowest-scoring task
            # (Eviction logic will be handled by the algorithm layer)
            return False

        # Write task to exploit pool (atomic operation)
        with self.exploit_tracker._backend.acquire_lock():
            # Find free slot in exploit pool
            exploit_index = self.exploit_tracker._next_free_index
            if exploit_index >= self.exploit_tracker.max_memory_tasks:
                return False  # No space

            # Copy all 18 float64 values
            exploit_data = self.exploit_tracker._backend.get_task_data(exploit_index)
            exploit_data[:] = task_data

            # Update exploit tracker's mapping
            self.exploit_tracker._task_id_to_index[task_id] = exploit_index

            # Find next free slot
            self.exploit_tracker._next_free_index = exploit_index + 1
            while self.exploit_tracker._next_free_index < self.exploit_tracker.max_memory_tasks:
                next_data = self.exploit_tracker._backend.get_task_data(self.exploit_tracker._next_free_index)
                if next_data[0] == 0.0:  # Slot is free
                    break
                self.exploit_tracker._next_free_index += 1

        # Remove from explore pool
        self.explore_tracker.remove_task(task_id)

        # Update pool map
        self._task_pool_map[task_id] = "exploit"

        return True

    def get_all_explore_tasks(self) -> List[int]:
        """Get all task IDs in the explore pool.

        Returns:
            List of task IDs in explore pool
        """
        return self.explore_tracker.get_all_tracked_tasks()

    def get_all_exploit_tasks(self) -> List[int]:
        """Get all task IDs in the exploit pool.

        Returns:
            List of task IDs in exploit pool
        """
        return self.exploit_tracker.get_all_tracked_tasks()

    def get_all_tracked_tasks(self) -> List[int]:
        """Get all task IDs from both pools combined.

        Returns:
            List of all task IDs across both explore and exploit pools
        """
        return self.get_all_explore_tasks() + self.get_all_exploit_tasks()

    @property
    def _total_completions(self) -> int:
        """Get total completions across both pools.

        Returns:
            Sum of completions from explore and exploit pools
        """
        return self.explore_tracker._total_completions + self.exploit_tracker._total_completions

    def get_global_stats(self) -> Dict[str, float]:
        """Get global performance statistics across both pools.

        Returns:
            Dictionary with combined statistics from explore and exploit pools
        """
        explore_stats = self.explore_tracker.get_global_stats()
        exploit_stats = self.exploit_tracker.get_global_stats()

        total_completions = explore_stats["total_completions"] + exploit_stats["total_completions"]

        if total_completions == 0:
            return {
                "mean_score": 0.0,
                "total_completions": 0,
                "explore_completions": 0,
                "exploit_completions": 0,
            }

        # Weighted average of mean scores
        explore_weight = explore_stats["total_completions"] / total_completions
        exploit_weight = exploit_stats["total_completions"] / total_completions
        combined_mean_score = (
            explore_stats["mean_score"] * explore_weight + exploit_stats["mean_score"] * exploit_weight
        )

        return {
            "mean_score": combined_mean_score,
            "total_completions": total_completions,
            "explore_completions": explore_stats["total_completions"],
            "exploit_completions": exploit_stats["total_completions"],
        }

    def update_task_performance(
        self,
        task_id: int,
        score: float,
        scorer: Optional[Any] = None,
        success_threshold: Optional[float] = None,
    ) -> None:
        """Update task performance in the appropriate pool.

        Args:
            task_id: Task to update
            score: Performance score
            scorer: Optional scorer for bidirectional EMAs
            success_threshold: Success threshold for binary classification
        """
        tracker = self.get_pool_tracker(task_id)
        if tracker is not None:
            tracker.update_task_performance_with_bidirectional_emas(task_id, score, scorer, success_threshold)

    def get_task_stats(self, task_id: int) -> Optional[Dict[str, float]]:
        """Get statistics for a specific task.

        Args:
            task_id: Task ID to query

        Returns:
            Task statistics dict, or None if not found
        """
        tracker = self.get_pool_tracker(task_id)
        if tracker is not None:
            return tracker.get_task_stats(task_id)
        return None

    def remove_task(self, task_id: int) -> None:
        """Remove a task from its pool.

        Args:
            task_id: Task to remove
        """
        tracker = self.get_pool_tracker(task_id)
        if tracker is not None:
            tracker.remove_task(task_id)
            del self._task_pool_map[task_id]

    def set_task_label(self, task_id: int, label: str) -> None:
        """Set label for a task in its pool.

        Args:
            task_id: Task ID
            label: Label string
        """
        tracker = self.get_pool_tracker(task_id)
        if tracker is not None:
            tracker.set_task_label(task_id, label)

    def get_task_label(self, task_id: int) -> Optional[str]:
        """Get label for a task.

        Args:
            task_id: Task ID

        Returns:
            Label string, or None if not found
        """
        tracker = self.get_pool_tracker(task_id)
        if tracker is not None:
            return tracker.get_task_label(task_id)
        return None

    def update_lp_score(self, task_id: int, lp_score: float) -> None:
        """Update the learning progress score for a task in its pool.

        Args:
            task_id: Task ID
            lp_score: New LP score
        """
        tracker = self.get_pool_tracker(task_id)
        if tracker is not None:
            tracker.update_lp_score(task_id, lp_score)

    def get_task_index(self, task_id: int) -> Optional[int]:
        """Get the index of a task within its pool.

        Args:
            task_id: Task ID

        Returns:
            Task index within its pool, or None if not found
        """
        tracker = self.get_pool_tracker(task_id)
        if tracker is not None:
            return tracker.get_task_index(task_id)
        return None

    def get_state(self) -> Dict[str, Any]:
        """Get dual-pool tracker state for checkpointing.

        Returns:
            Dictionary containing state of both pools
        """
        return {
            "explore_tracker": self.explore_tracker.get_state(),
            "exploit_tracker": self.exploit_tracker.get_state(),
            "task_pool_map": self._task_pool_map.copy(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load dual-pool tracker state from checkpoint.

        Args:
            state: State dictionary from get_state()
        """
        self.explore_tracker.load_state(state["explore_tracker"])
        self.exploit_tracker.load_state(state["exploit_tracker"])
        self._task_pool_map = state["task_pool_map"].copy()

    def cleanup_shared_memory(self) -> None:
        """Clean up shared memory for both pools."""
        self.explore_tracker.cleanup_shared_memory()
        self.exploit_tracker.cleanup_shared_memory()

    def __getstate__(self):
        """Prepare for pickling."""
        return {
            "num_explore_tasks": self.num_explore_tasks,
            "num_exploit_tasks": self.num_exploit_tasks,
            "explore_tracker": self.explore_tracker,
            "exploit_tracker": self.exploit_tracker,
            "task_pool_map": self._task_pool_map.copy(),
        }

    def __setstate__(self, state):
        """Restore from pickle."""
        self.num_explore_tasks = state["num_explore_tasks"]
        self.num_exploit_tasks = state["num_exploit_tasks"]
        self.explore_tracker = state["explore_tracker"]
        self.exploit_tracker = state["exploit_tracker"]
        self._task_pool_map = state["task_pool_map"]
