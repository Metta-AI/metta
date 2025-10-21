# Curriculum System Refactor Implementation Guide

**Purpose**: This document provides a comprehensive guide to reimplementing the curriculum refactor changes from the `msb_currcent_mergemain` branch onto a fresh branch from `main`. The goal is to avoid merge conflicts by providing step-by-step implementation instructions.

**Branch**: `msb_currcent_mergemain`
**Base**: `main`
**Focus**: Curriculum system refactor (not recipe changes)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Core Changes](#core-changes)
4. [Implementation Steps](#implementation-steps)
5. [File-by-File Changes](#file-by-file-changes)
6. [Configuration Changes](#configuration-changes)
7. [Testing Strategy](#testing-strategy)
8. [Migration Guide](#migration-guide)

---

## Executive Summary

### What Changed

The curriculum system underwent a major refactor to improve modularity, testability, and performance:

1. **Separation of Concerns**: Task tracking, scoring, and algorithm logic are now independent components
2. **Strategy Pattern**: Multiple implementations of task scoring (Bidirectional vs Basic LP)
3. **Memory Backend Abstraction**: Unified interface for local vs shared memory
4. **Statistics System**: Centralized stats collection with caching
5. **Improved Performance**: Better caching, reduced redundant computation
6. **Better Testability**: Each component can be tested in isolation

### Key Benefits

- ✅ Clean separation between task tracking and algorithm logic
- ✅ Easy to add new curriculum algorithms without modifying core code
- ✅ Identical code paths for single-process and multi-process training
- ✅ Comprehensive statistics with efficient caching
- ✅ Better performance through reduced lock contention
- ✅ Improved testing with mockable components

### Breaking Changes

**Configuration Parameters** (renamed or moved):
- `max_memory_tasks` → Now derived from `num_active_tasks` in `LearningProgressConfig`
- Default `num_active_tasks`: `16` → `1000`
- Default `ema_timescale`: `0.001` → `0.1`
- Default `rand_task_rate`: `0.25` → `0.01`
- New parameter: `use_shared_memory` (default: `True`)
- New parameter: `eviction_threshold_percentile` (default: `0.4`)

**Code Structure**:
- `TaskTracker` now unified (no separate Local/Centralized classes)
- Learning progress scoring moved to separate `LPScorer` classes
- Statistics moved to `stats.py`

---

## Architecture Overview

### Before Refactor

```
Curriculum
└── LearningProgressAlgorithm
    ├── Task tracking mixed with algorithm logic
    ├── Conditional code for centralized vs local
    ├── Learning progress calculations inline
    └── Limited extensibility
```

### After Refactor

```
Curriculum
└── CurriculumAlgorithm (abstract)
    ├── DiscreteRandomCurriculum (simple)
    └── LearningProgressAlgorithm
        ├── TaskTracker (unified)
        │   └── TaskMemoryBackend (strategy)
        │       ├── LocalMemoryBackend
        │       └── SharedMemoryBackend
        ├── LPScorer (strategy)
        │   ├── BidirectionalLPScorer
        │   └── BasicLPScorer
        ├── LPStatsAggregator
        └── CacheCoordinator
```

### Key Design Patterns

1. **Strategy Pattern**: Algorithm, Scorer, Backend are swappable
2. **Template Method**: Base classes define structure, subclasses implement details
3. **Facade Pattern**: StatsLogger provides unified interface to statistics
4. **Observer Pattern**: CacheCoordinator coordinates invalidation across components

---

## Core Changes

### 1. New File: `shared_memory_backend.py`

**Purpose**: Abstracts memory storage for task tracking

**Components**:
- `TaskMemoryBackend` - Abstract interface for task storage
- `LocalMemoryBackend` - In-memory storage for single-process
- `SharedMemoryBackend` - Shared memory for multi-process

**Key Features**:
- Unified interface: identical code for local/shared memory
- Thread-safe operations with locking
- Configurable structure sizes
- Session-based namespacing for shared memory

**Data Structure** (13 fields per task):
```python
[
    0:  task_id,
    1:  creation_time,
    2:  completion_count,
    3:  reward_ema,
    4:  lp_score,
    5:  success_rate_ema,
    6:  total_score,
    7:  last_score,
    8:  success_threshold,
    9:  seed,
    10: generator_type,
    11: ema_squared,  # NEW: for variance calculation
    12: is_active
]
```

### 2. New File: `lp_scorers.py`

**Purpose**: Separate scoring strategies from algorithm logic

**Components**:
- `LPScorer` - Abstract base class for scoring strategies
- `BidirectionalLPScorer` - Fast/slow EMA approach (default)
- `BasicLPScorer` - Variance-based approach (legacy)

**Key Changes**:
- Scoring logic extracted from `LearningProgressAlgorithm`
- Each scorer manages its own state
- Caching handled at scorer level
- Easy to add new scoring algorithms

### 3. Refactored: `task_tracker.py`

**Major Changes**:

**Before**:
```python
# Separate classes
class LocalTaskTracker:
    def __init__(self):
        self._task_memory = {}  # Dict-based

class CentralizedTaskTracker:
    def __init__(self, session_id):
        self._shared_mem = ...  # Shared memory
```

**After**:
```python
# Unified implementation
class TaskTracker:
    def __init__(self, backend=None, use_shared_memory=False):
        if backend is None:
            if use_shared_memory:
                backend = SharedMemoryBackend(...)
            else:
                backend = LocalMemoryBackend(...)
        self._backend = backend
```

**Key Improvements**:
- Single implementation for both local and shared memory
- Backend handles storage details
- Cleaner API with consistent behavior
- Better locking strategy (acquire only when needed)
- Added `ema_squared` tracking for variance calculations

**New Methods**:
- `update_lp_score(task_id, lp_score)` - Update LP score separately
- `cleanup_shared_memory()` - Clean up shared memory resources

**Enhanced Stats**:
```python
# Old stats
{
    "completion_count": int,
    "mean_score": float,
    "last_score": float,
    "age_seconds": float,
}

# New stats (additional fields)
{
    "reward_ema": float,          # NEW
    "ema_squared": float,         # NEW: for variance
    "lp_score": float,            # NEW
    "success_rate_ema": float,    # NEW
    "success_threshold": float,   # NEW
    "seed": float,                # NEW
    "generator_type": float,      # NEW
}
```

### 4. New File: `stats.py`

**Purpose**: Centralized statistics management

**Components**:

#### `StatsLogger` (Abstract Base)
- Unified interface for statistics across all components
- Base stats (always computed)
- Detailed stats (expensive, cached)
- Prefix support for namespacing

#### `SliceAnalyzer`
- Tracks task completion by slice values (bucketed parameters)
- Computes mean/variance per slice
- Identifies difficult parameter regions
- Configurable max axes for performance

#### `LPStatsAggregator`
- Aggregates stats from tracker, scorer, analyzer
- Avoids duplicate computation
- Consistent stat formatting
- Centralizes learning progress statistics

#### `CacheCoordinator`
- Centralizes cache invalidation logic
- Invalidates across stats, scorer, analyzer
- Triggered by task state changes
- Reduces code duplication

**Statistics Hierarchy**:
```
stats/
├── num_tasks
├── tracker/
│   ├── mean_recent_score
│   └── ...
├── lp/
│   ├── mean_learning_progress
│   ├── mean_task_success_rate
│   ├── mean_sample_prob
│   └── ...
└── slice/
    ├── game.num_plants_5/mean
    ├── game.num_plants_10/mean
    └── ...
```

### 5. Refactored: `learning_progress_algorithm.py`

**Major Simplifications**:

**Before** (~700 lines with mixed concerns):
```python
class LearningProgressAlgorithm:
    def __init__(...):
        # Task tracking
        self._task_memory = {}
        # Learning progress
        self._outcomes = {}
        self._p_fast = None
        self._p_slow = None
        # Statistics
        self._slice_data = {}
        # Caching
        self._stats_cache = {}

    def _update_bidirectional_progress(self):
        # ~150 lines of EMA math

    def _calculate_task_distribution(self):
        # ~50 lines of probability calculation

    def get_detailed_stats(self):
        # ~100 lines of stats aggregation
```

**After** (~300 lines with clear delegation):
```python
class LearningProgressAlgorithm:
    def __init__(...):
        # Delegate to components
        self.task_tracker = TaskTracker(...)
        self.scorer = BidirectionalLPScorer(...) if use_bidirectional else BasicLPScorer(...)
        self.stats_aggregator = LPStatsAggregator(...)
        self.cache_coordinator = CacheCoordinator(...)

    def score_tasks(self, task_ids):
        # Simple delegation
        return {tid: self.scorer.score_task(tid, self.task_tracker) for tid in task_ids}

    def update_task_performance(self, task_id, score):
        # Clear workflow
        self.scorer.update_with_score(task_id, score)
        lp_score = self.scorer.score_task(task_id, self.task_tracker)
        self.task_tracker.update_task_performance(task_id, score, lp_score=lp_score)
        self.cache_coordinator.invalidate_stats_cache()
```

**Configuration Changes**:
```python
class LearningProgressConfig:
    # NEW: Explicit configuration for all components
    num_active_tasks: int = 1000  # Changed from 16
    ema_timescale: float = 0.1    # Changed from 0.001
    slow_timescale_factor: float = 0.2  # NEW
    rand_task_rate: float = 0.01   # Changed from 0.25
    eviction_threshold_percentile: float = 0.4  # NEW

    # NEW: Task tracker configuration
    task_tracker_ema_alpha: float = 0.02
    task_struct_size: int = 13
    completion_history_size: int = 1000

    # NEW: Backend configuration
    use_shared_memory: bool = True
    session_id: Optional[str] = None

    # NEW: Basic mode parameters
    basic_ema_initial_alpha: float = 0.3
    basic_ema_alpha_decay: float = 0.2
    exploration_blend_factor: float = 0.5
```

### 6. Refactored: `task_generator.py`

**New Feature**: Task invariant validation

**Purpose**: Ensure all generated tasks have consistent:
- Number of resources
- Number of actions
- Number of agents

**Why**: Prevents evaluation using different environment configurations than training

**Implementation**:
```python
class TaskGenerator:
    def __init__(self, config):
        self._reference_task = None  # NEW: Store first task for comparison

    def get_task(self, task_id):
        mg_config = self._generate_task(task_id, rng)
        self._validate_task_invariants(mg_config, task_id)  # NEW
        return mg_config

    def _validate_task_invariants(self, mg_config, task_id):
        """Ensure critical environment parameters don't change across tasks."""
        if self._reference_task is None:
            self._reference_task = mg_config
            return

        # Check resource count
        assert len(mg_config.game.resource_names) == len(self._reference_task.game.resource_names), ...

        # Check action count
        assert count_actions(mg_config) == count_actions(self._reference_task), ...

        # Check agent count
        assert mg_config.game.num_agents == self._reference_task.game.num_agents, ...
```

### 7. Enhanced: `curriculum.py`

**Minimal Changes** (by design):
- Core curriculum logic remains stable
- Now works with any `CurriculumAlgorithm` implementation
- Better statistics integration via `StatsLogger`

**Key Addition**: Algorithm factory
```python
def _create_algorithm(config, num_tasks):
    if config.algorithm_config is None:
        return DiscreteRandomCurriculum(num_tasks, config.algorithm_config)
    elif config.algorithm_config.algorithm_type() == "learning_progress":
        return LearningProgressAlgorithm(num_tasks, config.algorithm_config)
    # Easy to add new algorithms here
```

### 8. New Documentation

**`structure.md`**:
- Complete architecture documentation
- Mermaid diagrams
- Data flow diagrams
- Design patterns explained
- Usage examples

**`sampling_with_replacement_example.md`**:
- Explains how curriculum sampling works
- Why multiple envs can run same task
- Probability distribution math
- Performance implications
- Example scenarios

---

## Implementation Steps

### Phase 1: Create New Infrastructure (No Breaking Changes)

**Step 1.1**: Create `shared_memory_backend.py`

```bash
touch metta/cogworks/curriculum/shared_memory_backend.py
```

**Implementation**: Copy the complete backend abstraction:
- `TaskMemoryBackend` abstract class
- `LocalMemoryBackend` implementation
- `SharedMemoryBackend` implementation

**Test**: Create `tests/cogworks/curriculum/test_shared_memory_backend.py`
- Test local backend operations
- Test shared memory backend operations
- Test backend switching
- Test process isolation

**Step 1.2**: Create `lp_scorers.py`

```bash
touch metta/cogworks/curriculum/lp_scorers.py
```

**Implementation**: Extract scoring logic:
- `LPScorer` abstract base class
- `BidirectionalLPScorer` (extract from `LearningProgressAlgorithm`)
- `BasicLPScorer` (extract basic mode logic)

**Test**: Create `tests/cogworks/curriculum/test_lp_scorers.py`
- Test scorer interface
- Test bidirectional scoring
- Test basic scoring
- Test score caching

**Step 1.3**: Create `stats.py`

```bash
touch metta/cogworks/curriculum/stats.py
```

**Implementation**: Extract statistics components:
- `StatsLogger` base class
- `SliceAnalyzer` (move from `learning_progress_algorithm.py`)
- `LPStatsAggregator` (new centralized aggregator)
- `CacheCoordinator` (new cache management)

**Test**: Create `tests/cogworks/curriculum/test_stats.py`
- Test stats logger interface
- Test slice analyzer
- Test stats aggregator
- Test cache coordinator

### Phase 2: Refactor Task Tracker (Careful Migration)

**Step 2.1**: Backup current implementation

```bash
cp metta/cogworks/curriculum/task_tracker.py metta/cogworks/curriculum/task_tracker_old.py
```

**Step 2.2**: Implement unified `TaskTracker`

**Changes**:
1. Remove separate `LocalTaskTracker` and `CentralizedTaskTracker` classes
2. Create single `TaskTracker` class that uses backend strategy
3. Add `_backend` attribute
4. Add `_task_id_to_index` mapping
5. Update all methods to use backend

**Key Method Changes**:

```python
# OLD
def track_task_creation(self, task_id: int) -> None:
    timestamp = time.time()
    self._task_memory[task_id] = (timestamp, 0, 0.0, 0.0)
    self._task_creation_order.append((timestamp, task_id))

# NEW
def track_task_creation(self, task_id: int, ...) -> None:
    with self._backend.acquire_lock():
        # Atomic operations on backend memory
        index = self._next_free_index
        self._task_id_to_index[task_id] = index
        task_data = self._backend.get_task_data(index)
        task_data[0] = float(task_id)
        task_data[1] = timestamp
        # ... initialize all fields
```

**Step 2.3**: Add backward compatibility factories

```python
def LocalTaskTracker(max_memory_tasks=1000, ema_alpha=0.1):
    """Factory for backward compatibility."""
    return TaskTracker(
        max_memory_tasks=max_memory_tasks,
        ema_alpha=ema_alpha,
        use_shared_memory=False
    )

def CentralizedTaskTracker(max_memory_tasks=1000, session_id=None, ema_alpha=0.1):
    """Factory for backward compatibility."""
    return TaskTracker(
        max_memory_tasks=max_memory_tasks,
        ema_alpha=ema_alpha,
        session_id=session_id,
        use_shared_memory=True
    )
```

**Step 2.4**: Update tests

Update `tests/cogworks/curriculum/test_task_tracker.py`:
- Test unified interface
- Test with local backend
- Test with shared backend
- Test backend switching
- Ensure existing tests still pass with factories

### Phase 3: Refactor Learning Progress Algorithm (Major Changes)

**Step 3.1**: Create new `LearningProgressConfig`

```python
class LearningProgressConfig(CurriculumAlgorithmConfig):
    """Updated configuration with new defaults and parameters."""

    # Core settings (updated defaults)
    num_active_tasks: int = 1000  # Was 16
    ema_timescale: float = 0.1    # Was 0.001
    rand_task_rate: float = 0.01   # Was 0.25

    # NEW: Explicit timescale configuration
    slow_timescale_factor: float = 0.2

    # NEW: Eviction control
    eviction_threshold_percentile: float = 0.4

    # NEW: Task tracker configuration
    task_tracker_ema_alpha: float = 0.02
    task_struct_size: int = 13
    completion_history_size: int = 1000
    task_default_success_threshold: float = 0.5
    task_default_generator_type: float = 0.0

    # NEW: Backend configuration
    use_shared_memory: bool = True
    session_id: Optional[str] = None

    # NEW: Basic mode parameters
    basic_ema_initial_alpha: float = 0.3
    basic_ema_alpha_decay: float = 0.2
    exploration_blend_factor: float = 0.5

    # Keep existing parameters
    use_bidirectional: bool = True
    exploration_bonus: float = 0.1
    progress_smoothing: float = 0.01
    performance_bonus_weight: float = 0.0
    sample_threshold: int = 10
    memory: int = 25
    max_slice_axes: int = 3
    enable_detailed_slice_logging: bool = False
```

**Step 3.2**: Refactor `__init__` method

```python
# OLD
def __init__(self, num_tasks, hypers):
    self.task_tracker = TaskTracker(max_memory_tasks=hypers.max_memory_tasks)
    # Initialize bidirectional or basic scoring inline
    if hypers.use_bidirectional:
        self._outcomes = {}
        self._p_fast = None
        # ... 20+ more attributes
    else:
        self._task_emas = {}
        # ... different attributes

# NEW
def __init__(self, num_tasks, hypers):
    # Unified task tracker with backend
    self.task_tracker = TaskTracker(
        max_memory_tasks=hypers.num_active_tasks,
        ema_alpha=hypers.task_tracker_ema_alpha,
        session_id=hypers.session_id if hypers.use_shared_memory else None,
        use_shared_memory=hypers.use_shared_memory,
        task_struct_size=hypers.task_struct_size,
        completion_history_size=hypers.completion_history_size,
        default_success_threshold=hypers.task_default_success_threshold,
        default_generator_type=hypers.task_default_generator_type,
    )

    # Scorer strategy (replaces inline logic)
    self.scorer = (
        BidirectionalLPScorer(hypers) if hypers.use_bidirectional
        else BasicLPScorer(hypers)
    )

    # Stats aggregator (replaces manual aggregation)
    self.stats_aggregator = LPStatsAggregator(
        task_tracker=self.task_tracker,
        scorer=self.scorer,
        slice_analyzer=self.slice_analyzer,
        num_tasks=num_tasks,
    )

    # Cache coordinator (replaces manual invalidation)
    self.cache_coordinator = CacheCoordinator(
        stats_logger=self,
        scorer=self.scorer,
        slice_analyzer=self.slice_analyzer,
    )
```

**Step 3.3**: Replace inline scoring methods

```python
# REMOVE: ~300 lines of inline scoring logic
# - _init_bidirectional_scoring()
# - _init_basic_scoring()
# - _update_bidirectional_progress()
# - _calculate_task_distribution()
# - _learning_progress()
# - _sigmoid()
# - _update_bidirectional_ema()
# - _update_basic_ema()
# - _get_bidirectional_learning_progress_score()
# - _get_basic_learning_progress_score()

# REPLACE WITH: Simple delegation
def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
    """Score tasks using the configured scorer strategy."""
    return {
        task_id: self.scorer.score_task(task_id, self.task_tracker)
        for task_id in task_ids
    }

def update_task_performance(self, task_id: int, score: float) -> None:
    """Update task performance using the scorer strategy."""
    # Update scorer's internal state
    self.scorer.update_with_score(task_id, score)

    # Calculate LP score from scorer
    lp_score = self.scorer.score_task(task_id, self.task_tracker)

    # Single atomic update to task tracker
    self.task_tracker.update_task_performance(
        task_id,
        score,
        lp_score=lp_score
    )

    # Track completion by label
    if task_id in self._task_labels:
        label = self._task_labels[task_id]
        self._label_completion_counts[label] = (
            self._label_completion_counts.get(label, 0) + 1
        )

    # Invalidate caches
    self.cache_coordinator.invalidate_stats_cache()
```

**Step 3.4**: Simplify eviction logic

```python
def should_evict_task(self, task_id: int, all_task_ids: List[int]) -> bool:
    """Check if task should be evicted based on learning progress."""
    task_stats = self.task_tracker.get_task_stats(task_id)
    if not task_stats or task_stats["completion_count"] < self.hypers.sample_threshold:
        return False

    # Score all tasks to find threshold
    scores = self.score_tasks(all_task_ids)
    task_score = scores.get(task_id, 0.0)

    # NEW: Configurable eviction threshold
    sorted_scores = sorted(scores.values())
    threshold_index = max(
        0,
        int(len(sorted_scores) * self.hypers.eviction_threshold_percentile)
    )
    threshold_score = sorted_scores[threshold_index] if sorted_scores else 0.0

    return task_score <= threshold_score
```

**Step 3.5**: Delegate statistics to aggregator

```python
# REMOVE: ~200 lines of manual stats computation
# - _get_bidirectional_detailed_stats()
# - _get_basic_detailed_stats()
# - Manual slice stats aggregation

# REPLACE WITH: Delegation
def get_detailed_stats(self) -> Dict[str, float]:
    """Get detailed stats from aggregator."""
    return self.stats_aggregator.get_all_stats(
        include_tracker=True,
        include_scorer=True,
        include_slice=self.hypers.enable_detailed_slice_logging
    )
```

**Step 3.6**: Update cleanup methods

```python
def on_task_evicted(self, task_id: int) -> None:
    """Clean up when a task is evicted."""
    # Task tracker handles its own locking
    self.task_tracker.remove_task(task_id)

    # Delegate to scorer
    self.scorer.remove_task(task_id)

    # Remove from label tracking
    self._task_labels.pop(task_id, None)

    # Invalidate caches via coordinator
    self.cache_coordinator.invalidate_stats_cache()
```

### Phase 4: Add Task Generator Validation

**Step 4.1**: Add validation to `TaskGenerator`

```python
class TaskGenerator:
    def __init__(self, config):
        self._config = config
        self._overrides = config.overrides
        self._reference_task = None  # NEW: Store first task

    def get_task(self, task_id: int) -> MettaGridConfig:
        """Generate a task with invariant validation."""
        rng = random.Random()
        rng.seed(task_id)
        mg_config = self._apply_overrides(
            self._generate_task(task_id, rng),
            self._config.overrides
        )

        # NEW: Validate invariants
        self._validate_task_invariants(mg_config, task_id)

        return mg_config
```

**Step 4.2**: Implement validation method

```python
def _validate_task_invariants(self, mg_config: MettaGridConfig, task_id: int) -> None:
    """Ensure critical environment parameters don't change across tasks.

    This validates that the number of resources, actions, and agents remain
    consistent across all tasks generated by this TaskGenerator.
    """
    if self._reference_task is None:
        # First task - establish reference
        self._reference_task = mg_config
        return

    ref = self._reference_task

    # Count enabled actions
    current_action_count = sum(
        1 for action in mg_config.game.actions.model_dump().values()
        if action.get("enabled", True)
    )
    ref_action_count = sum(
        1 for action in ref.game.actions.model_dump().values()
        if action.get("enabled", True)
    )

    # Validate resource count
    assert len(mg_config.game.resource_names) == len(ref.game.resource_names), (
        f"TaskGenerator produced inconsistent resource count for task {task_id}: "
        f"expected {len(ref.game.resource_names)}, got {len(mg_config.game.resource_names)}. "
        f"Resources must remain constant across all curriculum tasks."
    )

    # Validate action count
    assert current_action_count == ref_action_count, (
        f"TaskGenerator produced inconsistent action count for task {task_id}: "
        f"expected {ref_action_count}, got {current_action_count}. "
        f"Actions must remain constant across all curriculum tasks."
    )

    # Validate agent count
    assert mg_config.game.num_agents == ref.game.num_agents, (
        f"TaskGenerator produced inconsistent agent count for task {task_id}: "
        f"expected {ref.game.num_agents}, got {mg_config.game.num_agents}. "
        f"Number of agents must remain constant across all curriculum tasks."
    )
```

**Step 4.3**: Add test for validation

Create `tests/cogworks/curriculum/test_curriculum_invariants.py`:
```python
def test_task_generator_validates_resource_consistency():
    """Test that task generator catches inconsistent resource counts."""
    # Test that validation catches changes in resources, actions, agents
    ...

def test_task_generator_allows_consistent_tasks():
    """Test that consistent tasks pass validation."""
    ...
```

### Phase 5: Update Tests

**Step 5.1**: Update existing tests

Files to update:
- `tests/cogworks/curriculum/test_curriculum_core.py`
- `tests/cogworks/curriculum/test_curriculum_algorithms.py`
- `tests/cogworks/curriculum/test_curriculum_checkpointing.py`
- `tests/cogworks/curriculum/test_curriculum_capacity_eviction.py`
- `tests/cogworks/curriculum/test_curriculum_env.py`

**Changes needed**:
1. Update config defaults (num_active_tasks, ema_timescale, etc.)
2. Replace `max_memory_tasks` with `num_active_tasks`
3. Add `use_shared_memory=False` for tests (faster)
4. Update expected statistics keys (new prefixes)

**Step 5.2**: Add new test files

Create:
- `tests/cogworks/curriculum/test_shared_memory_backend.py` - Test memory backends
- `tests/cogworks/curriculum/test_lp_scorers.py` - Test scorer strategies
- `tests/cogworks/curriculum/test_curriculum_invariants.py` - Test task validation
- `tests/cogworks/curriculum/test_lp_config_overrides.py` - Test config overrides
- `tests/cogworks/curriculum/test_curriculum_shared_memory.py` - Test multi-process

### Phase 6: Add Documentation

**Step 6.1**: Create architecture documentation

```bash
touch metta/cogworks/curriculum/structure.md
```

Copy complete architecture documentation including:
- Component diagram (Mermaid)
- Design principles
- Data flow diagrams
- Usage examples
- Testing strategy

**Step 6.2**: Create sampling documentation

```bash
touch metta/cogworks/curriculum/sampling_with_replacement_example.md
```

Explain:
- How sampling with replacement works
- Why multiple envs can run same task
- Probability distribution
- Performance benefits

### Phase 7: Update Configuration Defaults

**Step 7.1**: Update recipe files (if needed)

Files that may need updates:
- `experiments/recipes/arena_basic_easy_shaped.py`
- `experiments/recipes/arena_with_sparse_rewards.py`
- Any recipes using curriculum

**Changes**:
```python
# OLD
curriculum_config = CurriculumConfig(
    algorithm_config=LearningProgressConfig(
        num_active_tasks=16,
        max_memory_tasks=1000,
    )
)

# NEW
curriculum_config = CurriculumConfig(
    algorithm_config=LearningProgressConfig(
        num_active_tasks=1000,  # Increased default
        use_shared_memory=True,  # Enabled by default
        # max_memory_tasks removed (derived from num_active_tasks)
    )
)
```

---

## File-by-File Changes

### New Files (Create These)

#### 1. `metta/cogworks/curriculum/shared_memory_backend.py` (~282 lines)

**Purpose**: Memory backend abstraction for task tracking

**Key Classes**:
- `TaskMemoryBackend(ABC)` - Abstract interface
- `LocalMemoryBackend` - NumPy arrays in process memory
- `SharedMemoryBackend` - Multiprocessing shared memory

**Structure**:
```python
class TaskMemoryBackend(ABC):
    """Abstract interface for task memory storage."""
    max_tasks: int
    task_struct_size: int
    completion_history_size: int

    @abstractmethod
    def get_task_data(self, index: int) -> np.ndarray: ...

    @abstractmethod
    def get_completion_history(self) -> np.ndarray: ...

    @abstractmethod
    def acquire_lock(self) -> ContextManager[Any]: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def cleanup(self) -> None: ...

class LocalMemoryBackend(TaskMemoryBackend):
    """In-memory backend for single-process use."""
    def __init__(self, max_tasks, task_struct_size=13, completion_history_size=1000):
        self._task_array = np.zeros((max_tasks, task_struct_size))
        self._completion_history = np.zeros((completion_history_size,))
        self._lock = RLock()
    # ... implementations

class SharedMemoryBackend(TaskMemoryBackend):
    """Shared memory backend for multi-process use."""
    def __init__(self, max_tasks, session_id=None, ...):
        # Create shared memory buffers
        self._task_shm = shared_memory.SharedMemory(
            name=f"curriculum_tasks_{session_id}",
            create=True,
            size=max_tasks * task_struct_size * 8
        )
        # ... process-safe locks
    # ... implementations
```

**Testing**: Test both backends with identical test suite to ensure consistent behavior

#### 2. `metta/cogworks/curriculum/lp_scorers.py` (~505 lines)

**Purpose**: Scoring strategies for learning progress

**Key Classes**:
- `LPScorer(ABC)` - Abstract scorer interface
- `BidirectionalLPScorer` - Fast/slow EMA scoring (default)
- `BasicLPScorer` - Variance-based scoring (legacy)

**Structure**:
```python
class LPScorer(ABC):
    """Abstract base class for learning progress scoring strategies."""
    def __init__(self, config: LearningProgressConfig):
        self.config = config

    @abstractmethod
    def score_task(self, task_id: int, tracker: TaskTracker) -> float:
        """Calculate learning progress score for a task."""
        ...

    @abstractmethod
    def update_with_score(self, task_id: int, score: float) -> None:
        """Update scorer state with new task completion."""
        ...

    @abstractmethod
    def remove_task(self, task_id: int) -> None:
        """Remove task from scorer."""
        ...

    @abstractmethod
    def get_stats(self) -> Dict[str, float]:
        """Get scorer statistics."""
        ...

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Serialize scorer state."""
        ...

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """Deserialize scorer state."""
        ...

    @abstractmethod
    def invalidate_cache(self) -> None:
        """Invalidate score cache."""
        ...

class BidirectionalLPScorer(LPScorer):
    """Bidirectional learning progress using fast and slow EMAs."""
    def __init__(self, config):
        super().__init__(config)
        self._outcomes: Dict[int, List[float]] = {}
        self._p_fast: Optional[np.ndarray] = None
        self._p_slow: Optional[np.ndarray] = None
        # ... state tracking

    def score_task(self, task_id, tracker):
        # Check cache
        if task_id in self._cache_valid_tasks:
            return self._score_cache[task_id]

        # Calculate LP from fast/slow EMA difference
        # ... scoring logic

        # Cache result
        self._score_cache[task_id] = score
        return score

    def _update_bidirectional_progress(self):
        """Update fast and slow EMAs for all tasks."""
        # ... EMA update logic

    def _calculate_task_distribution(self):
        """Calculate sampling distribution from LP scores."""
        # ... probability calculation

class BasicLPScorer(LPScorer):
    """Basic learning progress using variance estimation."""
    def score_task(self, task_id, tracker):
        # Use TaskTracker's EMAs for variance
        task_stats = tracker.get_task_stats(task_id)
        ema_score = task_stats["reward_ema"]
        ema_squared = task_stats["ema_squared"]

        # Calculate variance
        variance = max(0.0, ema_squared - (ema_score * ema_score))
        return np.sqrt(variance)
```

**Key Difference**: Bidirectional tracks its own state; Basic uses TaskTracker's EMAs

#### 3. `metta/cogworks/curriculum/stats.py` (~357 lines)

**Purpose**: Centralized statistics management

**Key Classes**:
- `StatsLogger(ABC)` - Base interface for stats
- `SliceAnalyzer` - Slice-based performance analysis
- `LPStatsAggregator` - Aggregates LP statistics
- `CacheCoordinator` - Manages cache invalidation

**Structure**:
```python
class StatsLogger(ABC):
    """Unified interface for statistics across curriculum components."""

    def __init__(self):
        self._stats_cache: Dict[str, float] = {}
        self._cache_valid = False

    @abstractmethod
    def get_base_stats(self) -> Dict[str, float]:
        """Get basic stats (always computed, fast)."""
        ...

    def get_detailed_stats(self) -> Dict[str, float]:
        """Get detailed stats (expensive, cached)."""
        if self._cache_valid:
            return self._stats_cache

        stats = self._compute_detailed_stats()
        self._stats_cache = stats
        self._cache_valid = True
        return stats

    @abstractmethod
    def _compute_detailed_stats(self) -> Dict[str, float]:
        """Compute detailed stats (override in subclass)."""
        ...

    def invalidate_cache(self) -> None:
        """Invalidate statistics cache."""
        self._cache_valid = False

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Get all stats with optional prefix."""
        all_stats = {**self.get_base_stats(), **self.get_detailed_stats()}
        if prefix:
            return {f"{prefix}/{k}": v for k, v in all_stats.items()}
        return all_stats

class SliceAnalyzer:
    """Analyzes task performance by slice values (bucketed parameters)."""

    def __init__(self, max_axes: int = 3):
        self.max_axes = max_axes
        self._slice_data: Dict[str, Dict[Any, List[float]]] = {}

    def update_task_completion(self, task_id, slice_values, score):
        """Record task completion with slice values."""
        for axis, value in slice_values.items():
            if len(self._slice_data) >= self.max_axes and axis not in self._slice_data:
                continue
            if axis not in self._slice_data:
                self._slice_data[axis] = {}
            if value not in self._slice_data[axis]:
                self._slice_data[axis][value] = []
            self._slice_data[axis][value].append(score)

    def get_base_stats(self):
        """Get counts of slices tracked."""
        return {
            "num_slice_axes": len(self._slice_data),
            "total_slice_values": sum(len(vals) for vals in self._slice_data.values()),
        }

    def get_detailed_stats(self):
        """Get mean/variance per slice value."""
        stats = {}
        for axis, values in self._slice_data.items():
            for value, scores in values.items():
                if scores:
                    stats[f"{axis}_{value}/mean"] = np.mean(scores)
                    stats[f"{axis}_{value}/std"] = np.std(scores)
        return stats

class LPStatsAggregator:
    """Aggregates statistics from tracker, scorer, and analyzer."""

    def __init__(self, task_tracker, scorer, slice_analyzer, num_tasks):
        self.task_tracker = task_tracker
        self.scorer = scorer
        self.slice_analyzer = slice_analyzer
        self.num_tasks = num_tasks

    def get_all_stats(self, include_tracker=True, include_scorer=True, include_slice=False):
        """Aggregate stats from all components."""
        stats = {"num_tasks": self.num_tasks}

        if include_tracker:
            tracker_stats = self.task_tracker.get_global_stats()
            for k, v in tracker_stats.items():
                stats[f"tracker/{k}"] = v

        if include_scorer:
            scorer_stats = self.scorer.get_stats()
            for k, v in scorer_stats.items():
                stats[f"lp/{k}"] = v

        if include_slice:
            slice_stats = self.slice_analyzer.get_detailed_stats()
            for k, v in slice_stats.items():
                stats[f"slice/{k}"] = v

        return stats

class CacheCoordinator:
    """Coordinates cache invalidation across components."""

    def __init__(self, stats_logger, scorer, slice_analyzer):
        self.stats_logger = stats_logger
        self.scorer = scorer
        self.slice_analyzer = slice_analyzer

    def invalidate_stats_cache(self):
        """Invalidate all caches when state changes."""
        self.stats_logger.invalidate_cache()
        self.scorer.invalidate_cache()
        # SliceAnalyzer doesn't cache currently
```

#### 4. `metta/cogworks/curriculum/structure.md` (~411 lines)

**Content**: Complete architecture documentation (see existing file)

**Sections**:
1. Overview
2. Architecture diagram (Mermaid)
3. Component descriptions
4. Design principles
5. Data flow diagrams
6. Key refactor changes
7. File organization
8. Usage examples
9. Testing approach

#### 5. `metta/cogworks/curriculum/sampling_with_replacement_example.md` (~296 lines)

**Content**: Explains sampling behavior (see existing file)

**Sections**:
1. Overview of sampling with replacement
2. How it works with vectorized envs
3. Probability distribution math
4. Benefits for curriculum learning
5. Comparison with/without replacement
6. Performance update flow
7. Shared memory coordination
8. Example scenarios

### Modified Files (Refactor These)

#### 1. `metta/cogworks/curriculum/task_tracker.py` (Major refactor)

**Changes Summary**:
- Remove separate Local/Centralized classes
- Create unified `TaskTracker` with backend strategy
- Add support for `ema_squared` tracking
- Enhanced statistics
- Better locking strategy

**Key Changes**:

```python
# OLD STRUCTURE (separate classes)
class LocalTaskTracker:
    def __init__(self, max_memory_tasks=1000):
        self._task_memory: Dict = {}

class CentralizedTaskTracker:
    def __init__(self, session_id, max_memory_tasks=1000):
        self._shared_memory = ...

# NEW STRUCTURE (unified with backend)
class TaskTracker:
    def __init__(self, max_memory_tasks=1000, ema_alpha=0.1,
                 backend=None, use_shared_memory=False, session_id=None,
                 task_struct_size=13, completion_history_size=1000,
                 default_success_threshold=0.5, default_generator_type=0.0):

        # Initialize or use provided backend
        if backend is None:
            if use_shared_memory:
                backend = SharedMemoryBackend(
                    max_tasks=max_memory_tasks,
                    session_id=session_id,
                    task_struct_size=task_struct_size,
                    completion_history_size=completion_history_size,
                )
            else:
                backend = LocalMemoryBackend(
                    max_tasks=max_memory_tasks,
                    task_struct_size=task_struct_size,
                    completion_history_size=completion_history_size,
                )

        self._backend = backend
        self._task_id_to_index: Dict[int, int] = {}
        self._next_free_index = 0
        self.ema_alpha = ema_alpha

        # Rebuild mapping from existing memory
        self._rebuild_task_mapping()
```

**Data Structure Changes**:

```python
# Task data structure (13 fields):
# [0]  task_id
# [1]  creation_time
# [2]  completion_count
# [3]  reward_ema
# [4]  lp_score           # NEW: stored here
# [5]  success_rate_ema   # NEW
# [6]  total_score
# [7]  last_score
# [8]  success_threshold  # NEW
# [9]  seed               # NEW
# [10] generator_type     # NEW
# [11] ema_squared        # NEW: for variance calculation
# [12] is_active          # NEW: soft deletion
```

**Method Changes**:

```python
# Enhanced update method
def update_task_performance(self, task_id, score, lp_score=None, success_threshold=None):
    """Update task performance with new completion score."""
    # Create task if needed
    if task_id not in self._task_id_to_index:
        self.track_task_creation(task_id, success_threshold=success_threshold or 0.5)

    with self._backend.acquire_lock():
        if task_id not in self._task_id_to_index:
            return  # Race condition

        index = self._task_id_to_index[task_id]
        task_data = self._backend.get_task_data(index)

        # Read current values
        completion_count = int(task_data[2])
        reward_ema = task_data[3]
        # ... more reads

        # Update EMAs
        new_reward_ema = (
            score if completion_count == 0
            else (1 - self.ema_alpha) * reward_ema + self.ema_alpha * score
        )

        # NEW: Update ema_squared for variance
        score_squared = score * score
        new_ema_squared = (
            score_squared if completion_count == 0
            else (1 - self.ema_alpha) * ema_squared + self.ema_alpha * score_squared
        )

        # Write updated values
        task_data[2] = float(completion_count + 1)
        task_data[3] = new_reward_ema
        task_data[4] = lp_score if lp_score is not None else task_data[4]
        # ... more writes
        task_data[11] = new_ema_squared  # NEW

# New method for updating LP score separately
def update_lp_score(self, task_id, lp_score):
    """Update the learning progress score for a task."""
    with self._backend.acquire_lock():
        if task_id not in self._task_id_to_index:
            return
        index = self._task_id_to_index[task_id]
        task_data = self._backend.get_task_data(index)
        task_data[4] = lp_score
```

**Backward Compatibility**:

```python
# Factory functions for backward compatibility
def LocalTaskTracker(max_memory_tasks=1000, ema_alpha=0.1):
    """Factory for backward compatibility."""
    return TaskTracker(
        max_memory_tasks=max_memory_tasks,
        ema_alpha=ema_alpha,
        use_shared_memory=False
    )

def CentralizedTaskTracker(max_memory_tasks=1000, session_id=None, ema_alpha=0.1):
    """Factory for backward compatibility."""
    return TaskTracker(
        max_memory_tasks=max_memory_tasks,
        ema_alpha=ema_alpha,
        session_id=session_id,
        use_shared_memory=True
    )
```

#### 2. `metta/cogworks/curriculum/learning_progress_algorithm.py` (Major simplification)

**Changes Summary**:
- Remove ~400 lines of inline scoring logic
- Delegate to `LPScorer` strategy
- Use `LPStatsAggregator` for statistics
- Use `CacheCoordinator` for cache management
- Simplified configuration

**Before** (~700 lines with mixed concerns)
**After** (~300 lines with clear delegation)

**Configuration Changes**:

```python
class LearningProgressConfig(CurriculumAlgorithmConfig):
    # UPDATED DEFAULTS
    num_active_tasks: int = 1000  # Was: 16
    ema_timescale: float = 0.1    # Was: 0.001
    rand_task_rate: float = 0.01   # Was: 0.25

    # NEW PARAMETERS
    slow_timescale_factor: float = 0.2
    eviction_threshold_percentile: float = 0.4
    task_tracker_ema_alpha: float = 0.02
    task_struct_size: int = 13
    completion_history_size: int = 1000
    task_default_success_threshold: float = 0.5
    task_default_generator_type: float = 0.0
    use_shared_memory: bool = True
    session_id: Optional[str] = None
    basic_ema_initial_alpha: float = 0.3
    basic_ema_alpha_decay: float = 0.2
    exploration_blend_factor: float = 0.5

    # KEPT FROM OLD
    use_bidirectional: bool = True
    exploration_bonus: float = 0.1
    progress_smoothing: float = 0.01  # Was: 0.05
    performance_bonus_weight: float = 0.0
    sample_threshold: int = 10
    memory: int = 25
    max_slice_axes: int = 3
    enable_detailed_slice_logging: bool = False

    # REMOVED
    # max_memory_tasks (now derived from num_active_tasks)
```

**Initialization Changes**:

```python
# OLD
def __init__(self, num_tasks, hypers):
    self.task_tracker = TaskTracker(max_memory_tasks=hypers.max_memory_tasks)

    # Inline initialization
    if hypers.use_bidirectional:
        self._outcomes = {}
        self._p_fast = None
        self._p_slow = None
        # ... 20+ attributes
    else:
        self._task_emas = {}
        # ... different attributes

# NEW
def __init__(self, num_tasks, hypers):
    # Unified task tracker
    self.task_tracker = TaskTracker(
        max_memory_tasks=hypers.num_active_tasks,
        ema_alpha=hypers.task_tracker_ema_alpha,
        session_id=hypers.session_id if hypers.use_shared_memory else None,
        use_shared_memory=hypers.use_shared_memory,
        task_struct_size=hypers.task_struct_size,
        completion_history_size=hypers.completion_history_size,
        default_success_threshold=hypers.task_default_success_threshold,
        default_generator_type=hypers.task_default_generator_type,
    )

    # Scorer strategy
    self.scorer = (
        BidirectionalLPScorer(hypers) if hypers.use_bidirectional
        else BasicLPScorer(hypers)
    )

    # Stats aggregator
    self.stats_aggregator = LPStatsAggregator(
        task_tracker=self.task_tracker,
        scorer=self.scorer,
        slice_analyzer=self.slice_analyzer,
        num_tasks=num_tasks,
    )

    # Cache coordinator
    self.cache_coordinator = CacheCoordinator(
        stats_logger=self,
        scorer=self.scorer,
        slice_analyzer=self.slice_analyzer,
    )

    # Label tracking
    self._task_labels: Dict[int, str] = {}
    self._label_completion_counts: Dict[str, int] = {}
```

**Method Simplifications**:

```python
# OLD: ~50 lines of conditional logic
def score_tasks(self, task_ids):
    if self.hypers.use_bidirectional:
        return self._score_tasks_bidirectional(task_ids)
    else:
        return self._score_tasks_basic(task_ids)

# NEW: Simple delegation
def score_tasks(self, task_ids):
    return {
        task_id: self.scorer.score_task(task_id, self.task_tracker)
        for task_id in task_ids
    }

# OLD: ~30 lines per method
def _update_bidirectional_ema(self, task_id, score): ...
def _update_basic_ema(self, task_id, score): ...

# NEW: Single method with delegation
def update_task_performance(self, task_id, score):
    self.scorer.update_with_score(task_id, score)
    lp_score = self.scorer.score_task(task_id, self.task_tracker)
    self.task_tracker.update_task_performance(task_id, score, lp_score=lp_score)

    # Track labels
    if task_id in self._task_labels:
        label = self._task_labels[task_id]
        self._label_completion_counts[label] = (
            self._label_completion_counts.get(label, 0) + 1
        )

    self.cache_coordinator.invalidate_stats_cache()

# OLD: ~100 lines of manual aggregation
def get_detailed_stats(self):
    # Manual computation from multiple sources
    ...

# NEW: Delegation to aggregator
def get_detailed_stats(self):
    return self.stats_aggregator.get_all_stats(
        include_tracker=True,
        include_scorer=True,
        include_slice=self.hypers.enable_detailed_slice_logging
    )
```

**Lines Removed** (~400 lines of inline logic):
- `_init_bidirectional_scoring()`
- `_init_basic_scoring()`
- `_update_bidirectional_progress()`
- `_calculate_task_distribution()`
- `_learning_progress()`
- `_sigmoid()`
- `_update_bidirectional_ema()`
- `_update_basic_ema()`
- `_get_bidirectional_learning_progress_score()`
- `_get_basic_learning_progress_score()`
- `_score_tasks_bidirectional()`
- `_score_tasks_basic()`
- `_get_bidirectional_detailed_stats()`
- `_get_basic_detailed_stats()`
- Manual stats aggregation logic

#### 3. `metta/cogworks/curriculum/task_generator.py` (Minor additions)

**Changes**: Add task invariant validation

**New Code**:

```python
class TaskGenerator:
    def __init__(self, config):
        self._config = config
        self._overrides = config.overrides
        self._reference_task = None  # NEW: Store first task for comparison

    def get_task(self, task_id: int) -> MettaGridConfig:
        """Generate a task (MettaGridConfig) using task_id as seed."""
        rng = random.Random()
        rng.seed(task_id)
        mg_config = self._apply_overrides(
            self._generate_task(task_id, rng),
            self._config.overrides
        )

        # NEW: Validate invariants across all generated tasks
        self._validate_task_invariants(mg_config, task_id)

        return mg_config

    def _validate_task_invariants(self, mg_config: MettaGridConfig, task_id: int) -> None:
        """Ensure critical environment parameters don't change across tasks.

        This validates that the number of resources, actions, and agents remain
        consistent across all tasks generated by this TaskGenerator. This prevents
        issues where evaluation uses different environment configurations than training.
        """
        if self._reference_task is None:
            # First task - establish reference invariants
            self._reference_task = mg_config
            return

        ref = self._reference_task

        # Get the number of enabled actions for both configs
        current_action_count = sum(
            1 for action in mg_config.game.actions.model_dump().values()
            if action.get("enabled", True)
        )
        ref_action_count = sum(
            1 for action in ref.game.actions.model_dump().values()
            if action.get("enabled", True)
        )

        # Validate resource count consistency
        assert len(mg_config.game.resource_names) == len(ref.game.resource_names), (
            f"TaskGenerator produced inconsistent resource count for task {task_id}: "
            f"expected {len(ref.game.resource_names)}, got {len(mg_config.game.resource_names)}. "
            f"Resources must remain constant across all curriculum tasks."
        )

        # Validate action count consistency
        assert current_action_count == ref_action_count, (
            f"TaskGenerator produced inconsistent action count for task {task_id}: "
            f"expected {ref_action_count}, got {current_action_count}. "
            f"Actions must remain constant across all curriculum tasks."
        )

        # Validate agent count consistency
        assert mg_config.game.num_agents == ref.game.num_agents, (
            f"TaskGenerator produced inconsistent agent count for task {task_id}: "
            f"expected {ref.game.num_agents}, got {mg_config.game.num_agents}. "
            f"Number of agents must remain constant across all curriculum tasks."
        )
```

**Default Change**:

```python
# OLD
def to_curriculum(self, num_active_tasks: int = 16, algorithm_config=None):
    ...

# NEW
def to_curriculum(self, num_active_tasks: int = 1000, algorithm_config=None):
    ...
```

#### 4. `metta/cogworks/curriculum/curriculum.py` (Minimal changes)

**Changes**: Minor updates for StatsLogger integration

**Key Changes**:
- Inherits from `StatsLogger` for consistent statistics interface
- Statistics now include prefixed data from algorithm
- No major logic changes (by design)

---

## Configuration Changes

### Parameter Mapping Table

| Old Parameter | New Parameter | Old Default | New Default | Notes |
|--------------|---------------|-------------|-------------|-------|
| `max_memory_tasks` | `num_active_tasks` | 1000 | 1000 | Now primary config param |
| `num_active_tasks` | `num_active_tasks` | 16 | 1000 | Increased for better curriculum |
| `ema_timescale` | `ema_timescale` | 0.001 | 0.1 | Better convergence |
| `rand_task_rate` | `rand_task_rate` | 0.25 | 0.01 | Less randomization |
| `progress_smoothing` | `progress_smoothing` | 0.05 | 0.01 | Less smoothing |
| N/A | `slow_timescale_factor` | N/A | 0.2 | NEW: Controls slow EMA |
| N/A | `eviction_threshold_percentile` | N/A | 0.4 | NEW: Eviction control |
| N/A | `task_tracker_ema_alpha` | N/A | 0.02 | NEW: Tracker EMA rate |
| N/A | `task_struct_size` | N/A | 13 | NEW: Memory structure |
| N/A | `use_shared_memory` | N/A | True | NEW: Enable shared memory |
| N/A | `session_id` | N/A | None | NEW: Shared memory ID |

### Configuration Migration

**Old Configuration**:
```python
curriculum_config = CurriculumConfig(
    task_generator=BucketedTaskGenerator.Config(...),
    algorithm_config=LearningProgressConfig(
        num_active_tasks=16,
        max_memory_tasks=1000,
        ema_timescale=0.001,
        rand_task_rate=0.25,
        use_bidirectional=True,
    )
)
```

**New Configuration** (with explicit defaults):
```python
curriculum_config = CurriculumConfig(
    task_generator=BucketedTaskGenerator.Config(...),
    algorithm_config=LearningProgressConfig(
        # Core parameters (updated defaults)
        num_active_tasks=1000,        # Increased from 16
        ema_timescale=0.1,            # Increased from 0.001
        rand_task_rate=0.01,          # Decreased from 0.25

        # New parameters with defaults
        slow_timescale_factor=0.2,
        eviction_threshold_percentile=0.4,
        task_tracker_ema_alpha=0.02,
        use_shared_memory=True,
        session_id=None,  # Auto-generated if None

        # Kept from old
        use_bidirectional=True,
        exploration_bonus=0.1,
        progress_smoothing=0.01,      # Decreased from 0.05
        performance_bonus_weight=0.0,
        sample_threshold=10,
        memory=25,
        max_slice_axes=3,
        enable_detailed_slice_logging=False,

        # Basic mode parameters (if use_bidirectional=False)
        basic_ema_initial_alpha=0.3,
        basic_ema_alpha_decay=0.2,
        exploration_blend_factor=0.5,
    )
)
```

**Minimal Configuration** (using new defaults):
```python
curriculum_config = CurriculumConfig(
    task_generator=BucketedTaskGenerator.Config(...),
    algorithm_config=LearningProgressConfig(
        # All other params use improved defaults
    )
)
```

---

## Testing Strategy

### Test Files Organization

#### New Test Files

1. **`tests/cogworks/curriculum/test_shared_memory_backend.py`**
   - Test `LocalMemoryBackend` operations
   - Test `SharedMemoryBackend` operations
   - Test backend switching
   - Test process isolation
   - Test lock behavior

2. **`tests/cogworks/curriculum/test_lp_scorers.py`**
   - Test `LPScorer` interface
   - Test `BidirectionalLPScorer` scoring logic
   - Test `BasicLPScorer` scoring logic
   - Test score caching
   - Test state serialization

3. **`tests/cogworks/curriculum/test_curriculum_invariants.py`**
   - Test task generation validation
   - Test resource count consistency
   - Test action count consistency
   - Test agent count consistency
   - Test error messages

4. **`tests/cogworks/curriculum/test_lp_config_overrides.py`**
   - Test configuration parameter overrides
   - Test default value changes
   - Test backward compatibility

5. **`tests/cogworks/curriculum/test_curriculum_shared_memory.py`**
   - Test multi-process coordination
   - Test shared memory synchronization
   - Test process-safe operations
   - Test cleanup

#### Updated Test Files

1. **`tests/cogworks/curriculum/test_curriculum_core.py`**
   - Update config defaults
   - Replace `max_memory_tasks` → `num_active_tasks`
   - Add `use_shared_memory=False` for faster tests
   - Update expected statistics keys

2. **`tests/cogworks/curriculum/test_curriculum_algorithms.py`**
   - Update LP config parameters
   - Test scorer delegation
   - Test stats aggregator
   - Test cache coordinator

3. **`tests/cogworks/curriculum/test_curriculum_checkpointing.py`**
   - Update state structure expectations
   - Test backend serialization
   - Test scorer state persistence
   - Test tracker state persistence

4. **`tests/cogworks/curriculum/test_curriculum_capacity_eviction.py`**
   - Update eviction threshold tests
   - Test configurable percentile
   - Test eviction with new LP scores

5. **`tests/cogworks/curriculum/test_curriculum_env.py`**
   - Update environment integration
   - Test sampling with replacement
   - Test multi-env coordination

### Test Migration Checklist

For each existing test:

- [ ] Replace `max_memory_tasks` with `num_active_tasks`
- [ ] Update `ema_timescale` if hardcoded (0.001 → 0.1)
- [ ] Update `rand_task_rate` if hardcoded (0.25 → 0.01)
- [ ] Add `use_shared_memory=False` for unit tests
- [ ] Update expected stat keys (`tracker/`, `lp/`, `slice/` prefixes)
- [ ] Update any direct references to scorer attributes
- [ ] Ensure tests work with both backend types

### Running Tests

```bash
# Run all curriculum tests
pytest tests/cogworks/curriculum/

# Run specific test file
pytest tests/cogworks/curriculum/test_shared_memory_backend.py

# Run with coverage
pytest --cov=metta.cogworks.curriculum tests/cogworks/curriculum/

# Run only new tests
pytest tests/cogworks/curriculum/test_shared_memory_backend.py \
       tests/cogworks/curriculum/test_lp_scorers.py \
       tests/cogworks/curriculum/test_curriculum_invariants.py \
       tests/cogworks/curriculum/test_lp_config_overrides.py \
       tests/cogworks/curriculum/test_curriculum_shared_memory.py
```

---

## Migration Guide

### For Users of Curriculum System

#### Updating Existing Code

**Step 1**: Update config parameters

```python
# OLD
config = LearningProgressConfig(
    num_active_tasks=16,
    max_memory_tasks=1000,
)

# NEW
config = LearningProgressConfig(
    num_active_tasks=1000,  # Or keep your value
    # max_memory_tasks removed (auto-derived)
)
```

**Step 2**: No code changes needed

The API remains the same:
- `curriculum.get_task()` - unchanged
- `curriculum.update_task_performance()` - unchanged
- `curriculum.stats()` - unchanged (keys may differ)

**Step 3**: Check statistics keys

If you're parsing specific stat keys:

```python
# OLD keys
stats["mean_task_success_rate"]
stats["mean_learning_progress"]

# NEW keys (with prefixes)
stats["lp/mean_task_success_rate"]
stats["lp/mean_learning_progress"]
stats["tracker/mean_recent_score"]
stats["slice/game.num_plants_5/mean"]
```

#### Benefits of Migration

✅ **Better Performance**:
- Reduced lock contention
- Better caching
- Faster statistics computation

✅ **Better Defaults**:
- `num_active_tasks=1000` (was 16)
- `ema_timescale=0.1` (was 0.001)
- Faster convergence and better curriculum

✅ **Easier Debugging**:
- Clearer statistics structure
- Better separation of concerns
- More consistent logging

### For Curriculum Developers

#### Adding New Scoring Algorithms

**Before**: Modify `LearningProgressAlgorithm` directly

**After**: Create new `LPScorer` subclass

```python
class MyCustomScorer(LPScorer):
    """My custom scoring algorithm."""

    def __init__(self, config):
        super().__init__(config)
        # Initialize custom state
        self._my_state = {}

    def score_task(self, task_id, tracker):
        """Calculate custom score."""
        task_stats = tracker.get_task_stats(task_id)
        # Your scoring logic
        return score

    def update_with_score(self, task_id, score):
        """Update custom state."""
        self._my_state[task_id] = score

    def remove_task(self, task_id):
        """Cleanup."""
        self._my_state.pop(task_id, None)

    def get_stats(self):
        """Report stats."""
        return {"custom_metric": ...}

    def get_state(self):
        """Serialize."""
        return {"my_state": self._my_state}

    def load_state(self, state):
        """Deserialize."""
        self._my_state = state["my_state"]

    def invalidate_cache(self):
        """Clear caches."""
        pass
```

#### Adding New Curriculum Algorithms

**Before**: Limited to `DiscreteRandomCurriculum` or `LearningProgressAlgorithm`

**After**: Extend `CurriculumAlgorithm` base class

```python
class MyCustomAlgorithm(CurriculumAlgorithm):
    """My custom curriculum algorithm."""

    def score_tasks(self, task_ids):
        """Score tasks for selection."""
        return {tid: my_score(tid) for tid in task_ids}

    def recommend_eviction(self, task_ids):
        """Recommend task to evict."""
        scores = self.score_tasks(task_ids)
        return min(scores, key=scores.get)

    def should_evict_task(self, task_id, all_task_ids):
        """Check if task should be evicted."""
        return my_eviction_logic(task_id)

    def update_task_performance(self, task_id, score):
        """Update with task completion."""
        # Your tracking logic
        pass

    def on_task_created(self, task):
        """Handle task creation."""
        pass

    def on_task_evicted(self, task_id):
        """Handle task eviction."""
        pass
```

#### Adding New Memory Backends

**Example**: Redis-backed storage

```python
class RedisMemoryBackend(TaskMemoryBackend):
    """Redis-backed memory for distributed training."""

    def __init__(self, max_tasks, redis_url, ...):
        self.max_tasks = max_tasks
        self.redis_client = redis.from_url(redis_url)
        self.lock = distributed_lock(redis_url)

    def get_task_data(self, index):
        """Get task data from Redis."""
        key = f"task:{index}"
        data = self.redis_client.get(key)
        return np.frombuffer(data, dtype=np.float64)

    def acquire_lock(self):
        """Acquire distributed lock."""
        return self.lock

    # ... implement other methods
```

---

## Summary

### Implementation Checklist

- [ ] **Phase 1**: Create new infrastructure files
  - [ ] `shared_memory_backend.py`
  - [ ] `lp_scorers.py`
  - [ ] `stats.py`
  - [ ] `structure.md`
  - [ ] `sampling_with_replacement_example.md`

- [ ] **Phase 2**: Refactor `task_tracker.py`
  - [ ] Create unified `TaskTracker` class
  - [ ] Add backend strategy pattern
  - [ ] Add backward compatibility factories
  - [ ] Update tests

- [ ] **Phase 3**: Refactor `learning_progress_algorithm.py`
  - [ ] Update `LearningProgressConfig`
  - [ ] Delegate to scorer strategy
  - [ ] Use stats aggregator
  - [ ] Use cache coordinator
  - [ ] Remove inline scoring logic (~400 lines)

- [ ] **Phase 4**: Update `task_generator.py`
  - [ ] Add task invariant validation
  - [ ] Update default `num_active_tasks`

- [ ] **Phase 5**: Update tests
  - [ ] Create new test files (5 files)
  - [ ] Update existing tests (10+ files)
  - [ ] Ensure all tests pass

- [ ] **Phase 6**: Add documentation
  - [ ] Architecture documentation
  - [ ] Sampling behavior documentation
  - [ ] Migration guide

- [ ] **Phase 7**: Update recipes (optional)
  - [ ] Update config defaults
  - [ ] Test with curriculum-based training

### Key Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| `learning_progress_algorithm.py` | ~700 lines | ~300 lines | -57% |
| Total curriculum code | ~1,500 lines | ~2,500 lines | +67%* |
| Testable components | 2 | 7 | +250% |
| Default `num_active_tasks` | 16 | 1000 | +6150% |
| Default `ema_timescale` | 0.001 | 0.1 | +10000% |

*Increase due to extracted components, but each is simpler and testable

### Benefits Summary

1. **Modularity**: Clear separation of concerns
2. **Testability**: Each component tests independently
3. **Extensibility**: Easy to add new algorithms/scorers
4. **Performance**: Better caching, less lock contention
5. **Maintainability**: Simpler code with clear responsibilities
6. **Documentation**: Comprehensive architecture docs
7. **Better Defaults**: Improved curriculum learning performance

---

## Questions & Troubleshooting

### Common Issues

**Q: Tests failing with "max_memory_tasks" not found**
A: Replace with `num_active_tasks` in config

**Q: Different statistics keys**
A: Statistics now use prefixes (`lp/`, `tracker/`, `slice/`)

**Q: Shared memory errors in tests**
A: Add `use_shared_memory=False` to test configs

**Q: Task validation errors**
A: Ensure task generator produces consistent resource/action/agent counts

**Q: Performance regression**
A: Check that shared memory is enabled for production (`use_shared_memory=True`)

### Debug Tips

```python
# Check backend type
print(curriculum.algorithm.task_tracker._backend.__class__.__name__)
# Should print: "SharedMemoryBackend" or "LocalMemoryBackend"

# Check scorer type
print(curriculum.algorithm.scorer.__class__.__name__)
# Should print: "BidirectionalLPScorer" or "BasicLPScorer"

# Get detailed stats
stats = curriculum.algorithm.stats_aggregator.get_all_stats(
    include_tracker=True,
    include_scorer=True,
    include_slice=True
)
print(json.dumps(stats, indent=2))
```

---

## Conclusion

This refactor represents a significant architectural improvement to the curriculum system while maintaining backward compatibility where possible. The modular design makes it easier to extend, test, and maintain the curriculum system going forward.

Key achievements:
- ✅ Clear separation of concerns
- ✅ Strategy pattern for algorithms and scorers
- ✅ Unified memory backend abstraction
- ✅ Comprehensive testing infrastructure
- ✅ Detailed documentation
- ✅ Improved performance through better caching
- ✅ Better defaults for curriculum learning

The implementation can be done incrementally, with each phase building on the previous one, minimizing risk and allowing for testing at each step.

