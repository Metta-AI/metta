"""Task performance tracking with pluggable memory backends.

This module provides task tracking infrastructure for curriculum learning. The implementation
works with both local memory (single-process) and shared memory (multi-process).

Core Classes:
    - TaskTracker: Task tracking with configurable memory backend

Key Responsibilities:
    - Track task creation and removal with O(1) lookup by task_id
    - Maintain exponential moving averages of task performance
    - Provide thread-safe access via backend-specific locking
    - Support state serialization for checkpointing

Memory Backend Abstraction:
    - LocalMemoryBackend: Fast numpy arrays for single-process training
    - SharedMemoryBackend: Multiprocessing shared memory for distributed workers
    - Backend selection is transparent to curriculum algorithms

Why Separate File:
    Task tracking is a distinct concern from curriculum logic. It manages the low-level
    storage and updates, while curriculum algorithms make high-level decisions about
    what to track and how to use the tracked data.

See Also:
    - learning_progress_algorithm.py: Uses TaskTracker for performance data
    - shared_memory_backend.py: Memory backend implementations
"""

import hashlib
import time
from typing import Any, Dict, List, Optional

from metta.cogworks.curriculum.shared_memory_backend import (
    LocalMemoryBackend,
    SharedMemoryBackend,
    TaskMemoryBackend,
    TaskState,
    TaskStateIndices,
)


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
        default_success_threshold: float = 0.5,
        default_generator_type: float = 0.0,
        enable_task_list_cache: bool = True,
    ):
        """Initialize task tracker with configurable backend.

        Args:
            max_memory_tasks: Maximum number of tasks to track
            ema_alpha: Alpha parameter for exponential moving average
            backend: Memory backend for task storage. If None, creates LocalMemoryBackend.
                    Config is responsible for creating and passing the appropriate backend.
            default_success_threshold: Default success threshold for new tasks (default: 0.5)
            default_generator_type: Default generator type identifier (default: 0.0)
            enable_task_list_cache: Enable caching of get_all_tracked_tasks() results
        """
        self.max_memory_tasks = max_memory_tasks
        self.ema_alpha = ema_alpha
        self.default_success_threshold = default_success_threshold
        self.default_generator_type = default_generator_type
        self.enable_task_list_cache = enable_task_list_cache

        # Running statistics for global tracking (replaces completion_history buffer)
        self._total_completions = 0
        self._sum_scores = 0.0

        # Initialize or use provided backend
        if backend is None:
            backend = LocalMemoryBackend(max_tasks=max_memory_tasks)

        self._backend: TaskMemoryBackend = backend
        self._task_id_to_index: Dict[int, int] = {}
        self._next_free_index = 0

        # Label tracking: hash -> label string mapping (local, not in shared memory)
        self._label_hash_to_string: Dict[int, str] = {}

        # Task list caching (for performance)
        self._cached_task_list: Optional[List[int]] = None
        self._task_list_cache_valid = False
        self._cache_hits = 0
        self._cache_misses = 0

        # Rebuild mapping from existing memory
        self._rebuild_task_mapping()

    def _rebuild_task_mapping(self) -> None:
        """Rebuild task ID to array index mapping by scanning backend memory.

        COLD PATH: Uses TaskState for initialization (called once at startup).
        """
        with self._backend.acquire_lock():
            self._task_id_to_index.clear()
            self._next_free_index = self._backend.max_tasks  # Default to end
            first_free_index = None

            for i in range(self._backend.max_tasks):
                state = self._backend.get_task_state(i)
                task_id = int(state.task_id)
                is_active = bool(state.is_active)

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
        """Track when a task is created with metadata.

        COLD PATH: Uses TaskState for type-safe initialization (called once per task).
        """
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

            # Initialize task state (type-safe, no magic offsets)
            state = TaskState(
                task_id=float(task_id),
                creation_time=timestamp,
                is_active=1.0,
                success_threshold=success_threshold,
                seed=float(seed),
                generator_type=generator_type,
                # All other fields default to 0.0:
                # completion_count, reward_ema, lp_score, success_rate_ema,
                # total_score, last_score, ema_squared, p_fast, p_slow,
                # p_true, random_baseline, label_hash
            )
            self._backend.set_task_state(index, state)

            # Find next free slot after this one
            self._next_free_index = index + 1
            while self._next_free_index < self._backend.max_tasks:
                next_state = self._backend.get_task_state(self._next_free_index)
                if next_state.task_id == 0.0:  # Slot is free
                    break
                self._next_free_index += 1

            # Invalidate task list cache (task pool changed)
            self._task_list_cache_valid = False

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

        Uses TaskState for type-safe read-modify-write (~1Hz frequency).
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

            # Read current state (type-safe)
            state = self._backend.get_task_state(index)
            completion_count = int(state.completion_count)

            # Update counts and totals
            state.completion_count = float(completion_count + 1)
            state.total_score += score
            state.last_score = score

            # Update reward EMA
            if completion_count == 0:
                state.reward_ema = score
            else:
                state.reward_ema = (1 - self.ema_alpha) * state.reward_ema + self.ema_alpha * score

            # Update EMA of squared scores (for variance calculation)
            score_squared = score * score
            if completion_count == 0:
                state.ema_squared = score_squared
            else:
                state.ema_squared = (1 - self.ema_alpha) * state.ema_squared + self.ema_alpha * score_squared

            # Update LP score if provided
            if lp_score is not None:
                state.lp_score = lp_score

            # Update success rate EMA
            current_threshold = success_threshold if success_threshold is not None else state.success_threshold
            state.success_threshold = current_threshold
            is_success = float(score >= current_threshold)
            if completion_count == 0:
                state.success_rate_ema = is_success
            else:
                state.success_rate_ema = (1 - self.ema_alpha) * state.success_rate_ema + self.ema_alpha * is_success

            # Write back updated state
            self._backend.set_task_state(index, state)

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

        PERFORMANCE: This is a HOT PATH method called at ~200Hz (every episode completion).
        Uses direct array access to avoid Pydantic overhead (~50x faster than TaskState).

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

            # FAST PATH: Direct array access (no Pydantic overhead)
            task_data = self._backend.get_task_data(index)
            completion_count = int(task_data[TaskStateIndices.COMPLETION_COUNT])

            # === PART 1: Basic EMA updates ===
            # Update counts and totals
            task_data[TaskStateIndices.COMPLETION_COUNT] = float(completion_count + 1)
            task_data[TaskStateIndices.TOTAL_SCORE] += score
            task_data[TaskStateIndices.LAST_SCORE] = score

            # Update reward EMA
            if completion_count == 0:
                task_data[TaskStateIndices.REWARD_EMA] = score
            else:
                reward_ema = task_data[TaskStateIndices.REWARD_EMA]
                task_data[TaskStateIndices.REWARD_EMA] = (1 - self.ema_alpha) * reward_ema + self.ema_alpha * score

            # Update EMA of squared scores
            score_squared = score * score
            if completion_count == 0:
                task_data[TaskStateIndices.EMA_SQUARED] = score_squared
            else:
                ema_squared = task_data[TaskStateIndices.EMA_SQUARED]
                task_data[TaskStateIndices.EMA_SQUARED] = (
                    1 - self.ema_alpha
                ) * ema_squared + self.ema_alpha * score_squared

            # Update success rate EMA
            current_threshold = (
                success_threshold if success_threshold is not None else task_data[TaskStateIndices.SUCCESS_THRESHOLD]
            )
            task_data[TaskStateIndices.SUCCESS_THRESHOLD] = current_threshold
            is_success = float(score >= current_threshold)
            if completion_count == 0:
                task_data[TaskStateIndices.SUCCESS_RATE_EMA] = is_success
            else:
                success_rate_ema = task_data[TaskStateIndices.SUCCESS_RATE_EMA]
                task_data[TaskStateIndices.SUCCESS_RATE_EMA] = (
                    1 - self.ema_alpha
                ) * success_rate_ema + self.ema_alpha * is_success

            # === PART 2: Bidirectional EMA updates (if scorer provided) ===
            if scorer is not None and hasattr(scorer, "config"):
                task_success_rate = score

                # Handle baseline normalization if enabled
                if scorer.config.use_baseline_normalization:
                    # Set baseline on first update (capped at 0.75)
                    random_baseline = task_data[TaskStateIndices.RANDOM_BASELINE]
                    if random_baseline == 0.0:
                        random_baseline = min(task_success_rate, 0.75)
                        task_data[TaskStateIndices.RANDOM_BASELINE] = random_baseline

                    # Calculate normalized "mastery" score
                    improvement_over_baseline = max(task_success_rate - random_baseline, 0.0)
                    total_possible_improvement = max(1.0 - random_baseline, 1e-10)
                    normalized_task_success_rate = improvement_over_baseline / total_possible_improvement
                else:
                    # Use raw success rate
                    normalized_task_success_rate = task_success_rate

                # Initialize or update bidirectional EMAs
                p_fast = task_data[TaskStateIndices.P_FAST]
                p_slow = task_data[TaskStateIndices.P_SLOW]

                if p_fast == 0.0 and p_slow == 0.0:
                    # First update - initialize to current value
                    task_data[TaskStateIndices.P_FAST] = normalized_task_success_rate
                    task_data[TaskStateIndices.P_SLOW] = normalized_task_success_rate
                    task_data[TaskStateIndices.P_TRUE] = task_success_rate
                else:
                    # Update EMAs
                    ema_timescale = scorer.config.ema_timescale
                    task_data[TaskStateIndices.P_FAST] = normalized_task_success_rate * ema_timescale + p_fast * (
                        1.0 - ema_timescale
                    )
                    slow_timescale = ema_timescale * scorer.config.slow_timescale_factor
                    task_data[TaskStateIndices.P_SLOW] = normalized_task_success_rate * slow_timescale + p_slow * (
                        1.0 - slow_timescale
                    )
                    p_true = task_data[TaskStateIndices.P_TRUE]
                    task_data[TaskStateIndices.P_TRUE] = task_success_rate * ema_timescale + p_true * (
                        1.0 - ema_timescale
                    )

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
                state = self._backend.get_task_state(i)
                if int(state.task_id) == task_id and bool(state.is_active):
                    return i

        return None

    def update_lp_score(self, task_id: int, lp_score: float) -> None:
        """Update the learning progress score for a task."""
        index = self.get_task_index(task_id)
        if index is None:
            return

        with self._backend.acquire_lock():
            state = self._backend.get_task_state(index)
            state.lp_score = lp_score
            self._backend.set_task_state(index, state)

    def update_bidirectional_emas(
        self, task_id: int, p_fast: float, p_slow: float, p_true: float, random_baseline: float
    ) -> None:
        """Update bidirectional EMA values for a task.

        Args:
            task_id: Task ID to update
            p_fast: Fast EMA value
            p_slow: Slow EMA value
            p_true: True performance EMA value
            random_baseline: Random baseline value
        """
        index = self.get_task_index(task_id)
        if index is None:
            return

        with self._backend.acquire_lock():
            state = self._backend.get_task_state(index)
            state.p_fast = p_fast
            state.p_slow = p_slow
            state.p_true = p_true
            state.random_baseline = random_baseline
            self._backend.set_task_state(index, state)

    def get_task_stats(self, task_id: int) -> Optional[Dict[str, float]]:
        """Get statistics for a specific task.

        For shared memory backends, will search shared memory if task is not in local mapping.
        This allows reading stats for tasks created by other workers.

        Note: No locking - may read slightly stale data, but that's acceptable
        for statistics queries to avoid lock contention.

        COLD PATH: Uses TaskState for type-safe field access (called for monitoring/debugging).
        """
        # Fast path: check local mapping first
        if task_id in self._task_id_to_index:
            index = self._task_id_to_index[task_id]
        elif isinstance(self._backend, SharedMemoryBackend):
            # Slow path: scan shared memory to find task from another worker
            index = None
            for i in range(self._backend.max_tasks):
                state = self._backend.get_task_state(i)
                if int(state.task_id) == task_id and bool(state.is_active):
                    index = i
                    break

            if index is None:
                return None
        else:
            # Local memory backend and task not found
            return None

        # Get task state (type-safe, no magic offsets!)
        state = self._backend.get_task_state(index)

        if state.is_active == 0:  # not active
            return None

        completion_count = int(state.completion_count)
        age_seconds = time.time() - state.creation_time

        if completion_count == 0:
            return {
                "completion_count": 0,
                "mean_score": 0.0,
                "reward_ema": 0.0,
                "ema_squared": 0.0,
                "lp_score": 0.0,
                "success_rate_ema": 0.0,
                "last_score": 0.0,
                "success_threshold": state.success_threshold,
                "seed": state.seed,
                "generator_type": state.generator_type,
                "age_seconds": age_seconds,
                "p_fast": state.p_fast,
                "p_slow": state.p_slow,
                "p_true": state.p_true,
                "random_baseline": state.random_baseline,
            }

        return {
            "completion_count": completion_count,
            "mean_score": state.total_score / completion_count,
            "reward_ema": state.reward_ema,
            "ema_squared": state.ema_squared,
            "lp_score": state.lp_score,
            "success_rate_ema": state.success_rate_ema,
            "last_score": state.last_score,
            "success_threshold": state.success_threshold,
            "seed": state.seed,
            "generator_type": state.generator_type,
            "age_seconds": age_seconds,
            "p_fast": state.p_fast,
            "p_slow": state.p_slow,
            "p_true": state.p_true,
            "random_baseline": state.random_baseline,
        }

    def get_all_tracked_tasks(self) -> List[int]:
        """Get all currently tracked task IDs.

        For shared memory backends, scans shared memory to find ALL tasks from ALL workers.
        For local memory backends, returns tasks from the local mapping.

        Note: No locking - returns snapshot which may be slightly stale.

        Performance: Results are cached to avoid repeated O(max_tasks) scans.
        Cache is invalidated on task creation/removal.
        """
        # Check cache first (if enabled)
        if self.enable_task_list_cache and self._task_list_cache_valid:
            self._cache_hits += 1
            return self._cached_task_list.copy() if self._cached_task_list else []

        self._cache_misses += 1

        # Log cache effectiveness periodically
        if self.enable_task_list_cache and (self._cache_hits + self._cache_misses) % 1000 == 0:
            hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses)
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"[LP_PERF] Task list cache: {self._cache_hits} hits, "
                f"{self._cache_misses} misses, hit rate: {hit_rate:.1%}"
            )

        # For shared memory, scan to see tasks from all workers
        if isinstance(self._backend, SharedMemoryBackend):
            task_ids = []
            for i in range(self._backend.max_tasks):
                state = self._backend.get_task_state(i)
                # A slot is active if is_active flag is True
                # (task_id == 0 with is_active == False means free slot)
                if bool(state.is_active):
                    task_ids.append(int(state.task_id))
        else:
            # Local memory: use mapping (only this process's tasks)
            task_ids = list(self._task_id_to_index.keys())

        # Update cache
        if self.enable_task_list_cache:
            self._cached_task_list = task_ids
            self._task_list_cache_valid = True

        return task_ids

    def remove_task(self, task_id: int) -> None:
        """Remove a task from tracking.

        COLD PATH: Uses TaskState for type-safe reset (called during eviction).
        """
        with self._backend.acquire_lock():
            if task_id in self._task_id_to_index:
                index = self._task_id_to_index[task_id]

                # Reset to empty state (type-safe, no magic offsets)
                empty_state = TaskState()  # All fields default to 0.0
                self._backend.set_task_state(index, empty_state)

                del self._task_id_to_index[task_id]

                # Update _next_free_index to enable slot reuse
                # If we just freed a slot before the current free index, update it
                if index < self._next_free_index:
                    self._next_free_index = index

                # Invalidate task list cache (task pool changed)
                self._task_list_cache_valid = False

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

        # Compute DETERMINISTIC hash for label using SHA256
        # This ensures consistent hashes across all processes (unlike Python's hash())
        # Use only 53 bits to ensure exact float64 representation (2^53 - 1)
        sha256_hash = hashlib.sha256(label.encode("utf-8")).digest()
        hash_int = int.from_bytes(sha256_hash[:8], byteorder="big")
        label_hash = hash_int & 0x1FFFFFFFFFFFFF  # 53-bit mask

        # Store hash in shared memory
        with self._backend.acquire_lock():
            index = self._task_id_to_index[task_id]
            state = self._backend.get_task_state(index)
            state.label_hash = float(label_hash)
            self._backend.set_task_state(index, state)

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
        state = self._backend.get_task_state(index)
        label_hash = int(state.label_hash)

        if label_hash == 0:
            return None  # No label set

        return self._label_hash_to_string.get(label_hash)

    def get_label_completion_counts(self) -> Dict[str, int]:
        """Count total completions per label by scanning ALL slots in shared memory.

        This scans all memory slots (not just local task mapping) to find tasks
        across all processes, ensuring cross-process label visibility.

        Returns:
            Dictionary mapping label -> total completion count
        """
        label_counts: Dict[str, int] = {}

        # Scan ALL slots in shared memory to find tasks from all processes
        for slot_index in range(self._backend.max_tasks):
            state = self._backend.get_task_state(slot_index)

            if not bool(state.is_active):
                continue

            label_hash = int(state.label_hash)
            if label_hash == 0:
                continue  # No label

            label = self._label_hash_to_string.get(label_hash)
            if label is None:
                continue  # Hash not in local mapping (from another process)

            completion_count = int(state.completion_count)
            label_counts[label] = label_counts.get(label, 0) + completion_count

        return label_counts

    def get_state(self) -> Dict[str, Any]:
        """Get task tracker state for checkpointing.

        Note: No locking - checkpoint may have minor inconsistencies if captured
        during updates, but this is acceptable as checkpoints are infrequent.
        """
        task_memory = {}
        for task_id, index in self._task_id_to_index.items():
            state = self._backend.get_task_state(index)
            if state.is_active > 0:
                task_memory[task_id] = {
                    "creation_time": state.creation_time,
                    "completion_count": int(state.completion_count),
                    "reward_ema": state.reward_ema,
                    "lp_score": state.lp_score,
                    "success_rate_ema": state.success_rate_ema,
                    "total_score": state.total_score,
                    "last_score": state.last_score,
                    "success_threshold": state.success_threshold,
                    "seed": state.seed,
                    "generator_type": state.generator_type,
                    "ema_squared": state.ema_squared,
                    "p_fast": state.p_fast,
                    "p_slow": state.p_slow,
                    "p_true": state.p_true,
                    "random_baseline": state.random_baseline,
                    "label_hash": state.label_hash,
                }

        total_completions = sum(
            int(self._backend.get_task_state(idx).completion_count) for idx in self._task_id_to_index.values()
        )

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

                # Create TaskState from checkpoint data (type-safe)
                restored_state = TaskState(
                    task_id=float(task_id),
                    creation_time=task_data.get("creation_time", time.time()),
                    completion_count=float(task_data.get("completion_count", 0)),
                    reward_ema=task_data.get("reward_ema", 0.0),
                    lp_score=task_data.get("lp_score", 0.0),
                    success_rate_ema=task_data.get("success_rate_ema", 0.0),
                    total_score=task_data.get("total_score", 0.0),
                    last_score=task_data.get("last_score", 0.0),
                    success_threshold=task_data.get("success_threshold", 0.5),
                    seed=task_data.get("seed", 0.0),
                    generator_type=task_data.get("generator_type", 0.0),
                    ema_squared=task_data.get("ema_squared", 0.0),
                    is_active=1.0,
                    p_fast=task_data.get("p_fast", 0.0),
                    p_slow=task_data.get("p_slow", 0.0),
                    p_true=task_data.get("p_true", 0.0),
                    random_baseline=task_data.get("random_baseline", 0.0),
                    label_hash=task_data.get("label_hash", 0.0),
                )
                self._backend.set_task_state(i, restored_state)

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
            # Performance optimization attributes
            "enable_task_list_cache": self.enable_task_list_cache,
            "cached_task_list": self._cached_task_list,
            "task_list_cache_valid": self._task_list_cache_valid,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
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

        # Restore performance optimization attributes (with defaults for backwards compatibility)
        self.enable_task_list_cache = state.get("enable_task_list_cache", True)
        self._cached_task_list = state.get("cached_task_list", None)
        self._task_list_cache_valid = state.get("task_list_cache_valid", False)
        self._cache_hits = state.get("cache_hits", 0)
        self._cache_misses = state.get("cache_misses", 0)

        # Initialize label tracking (local per-process)
        self._label_hash_to_string = {}

        # Rebuild task ID to index mapping from shared memory
        self._task_id_to_index = {}
        self._next_free_index = 0
        self._rebuild_task_mapping()


def create_task_tracker(
    max_memory_tasks: int,
    ema_alpha: float,
    use_shared_memory: bool,
    default_success_threshold: float,
    default_generator_type: float,
    _session_id: Optional[str] = None,
    enable_task_list_cache: bool = True,
) -> TaskTracker:
    """Create a TaskTracker with appropriate backend based on configuration.

    This factory function encapsulates backend selection logic so callers don't need
    to know about backend classes. The task struct size is automatically computed from
    TaskState.struct_size().

    All parameters (except _session_id) are required and should come from config.
    Defaults are defined in LearningProgressConfig, not here.

    Args:
        max_memory_tasks: Maximum number of tasks to track
        ema_alpha: Alpha parameter for exponential moving average
        use_shared_memory: If True, creates SharedMemoryBackend; else LocalMemoryBackend
        default_success_threshold: Default success threshold for new tasks
        default_generator_type: Default generator type identifier
        _session_id: Internal - session ID for shared memory. Auto-generated by config.
        enable_task_list_cache: Enable caching of get_all_tracked_tasks() for performance

    Returns:
        TaskTracker instance with appropriate backend
    """
    if use_shared_memory:
        backend = SharedMemoryBackend(
            max_tasks=max_memory_tasks,
            session_id=_session_id,
        )
    else:
        backend = LocalMemoryBackend(
            max_tasks=max_memory_tasks,
        )

    return TaskTracker(
        max_memory_tasks=max_memory_tasks,
        ema_alpha=ema_alpha,
        backend=backend,
        default_success_threshold=default_success_threshold,
        default_generator_type=default_generator_type,
        enable_task_list_cache=enable_task_list_cache,
    )
