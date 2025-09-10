# Learning Progress Pre-Dehydration State

This document captures the exact implementation of learning progress curriculum as it existed before the dehydration process (commit `4167f6684` - "initial minimal functionality").

## Architecture Overview

The pre-dehydration system had **two parallel implementations** serving different use cases:

1. **Minimal Learning Progress** (`metta/cogworks/curriculum/learning_progress_minimal.py`) - Simple task-level tracking
2. **Sophisticated Learning Progress** (`mettagrid/src/metta/mettagrid/curriculum/learning_progress.py`) - Fixed-space bidirectional tracking

## Implementation 1: Minimal Learning Progress

**File**: `metta/cogworks/curriculum/learning_progress_minimal.py`

### Configuration
```python
class LearningProgressCurriculumConfig(BaseModel):
    ema_timescale: float = Field(default=0.001, gt=0, le=1.0)
    progress_smoothing: float = Field(default=0.05, ge=0, le=1.0)
    rand_task_rate: float = Field(default=0.25, ge=0, le=1.0)
    memory: int = Field(default=25, gt=0)
```

### Task Tracking
```python
class LearningProgressCurriculumTask:
    def __init__(self, config, task_id, env_cfg):
        self._config = config
        self._task_id = task_id
        self._env_cfg = env_cfg
        self._outcomes: List[float] = []
        self._p_fast: float = 0.0
        self._p_slow: float = 0.0
        self._initialized: bool = False

    def complete(self, score: float):
        # Store clipped outcome
        clipped_score = max(0.0, min(1.0, score))
        self._outcomes.append(clipped_score)

        # Respect memory limit
        if len(self._outcomes) > self._config.memory:
            self._outcomes = self._outcomes[-self._config.memory:]

        self._update_learning_progress()

    def _update_learning_progress(self):
        if not self._outcomes:
            return

        success_rate = float(np.mean(self._outcomes))

        if not self._initialized:
            self._p_fast = success_rate
            self._p_slow = success_rate
            self._initialized = True
        else:
            # Update EMA trackers
            self._p_fast = float(
                success_rate * self._config.ema_timescale + 
                self._p_fast * (1.0 - self._config.ema_timescale)
            )
            self._p_slow = float(
                self._p_fast * self._config.ema_timescale + 
                self._p_slow * (1.0 - self._config.ema_timescale)
            )

    def get_learning_progress(self) -> float:
        if not self._initialized:
            return 0.0
        return abs(self._p_fast - self._p_slow)
```

### Curriculum Management
```python
class LearningProgressCurriculum:
    def __init__(self, config, seed=0):
        self._config = config
        self._rng = np.random.RandomState(seed)
        self._tasks: List[LearningProgressCurriculumTask] = []
        self._task_weights: List[float] = []

    def get_task(self) -> LearningProgressCurriculumTask:
        if not self._tasks:
            # Create initial task
            task = LearningProgressCurriculumTask(self._config, 0, {})
            self._tasks.append(task)
            return task

        # Simple weighted sampling based on learning progress
        if self._task_weights:
            task_idx = self._rng.choice(len(self._tasks), p=self._task_weights)
            return self._tasks[task_idx]
        else:
            return self._tasks[self._rng.randint(0, len(self._tasks))]

    def _update_weights(self):
        if not self._tasks:
            return

        # Get learning progress for each task
        learning_progress = [task.get_learning_progress() for task in self._tasks]

        # Simple normalization
        total = sum(learning_progress) + 1e-6
        self._task_weights = [lp / total for lp in learning_progress]
```

**Characteristics:**
- **Complexity**: O(n) where n = number of task instances
- **Memory**: Bounded per task (25 outcomes default)
- **Task Management**: Dynamic list of task instances
- **Sampling**: Weighted random selection from task list

## Implementation 2: Sophisticated Learning Progress (MettagGrid)

**File**: `mettagrid/src/metta/mettagrid/curriculum/learning_progress.py`

### Main Curriculum Class
```python
class LearningProgressCurriculum(RandomCurriculum):
    def __init__(self, tasks, env_overrides=None, ema_timescale=0.001, 
                 progress_smoothing=0.05, num_active_tasks=16, rand_task_rate=0.25, 
                 sample_threshold=10, memory=25):
        super().__init__(tasks, env_overrides)

        # Initialize learning progress tracker
        search_space_size = len(tasks)
        self._lp_tracker = BidirectionalLearningProgress(
            search_space=search_space_size,
            ema_timescale=ema_timescale,
            progress_smoothing=progress_smoothing,
            num_active_tasks=num_active_tasks,
            rand_task_rate=rand_task_rate,
            sample_threshold=sample_threshold,
            memory=memory,
        )

    def complete_task(self, id: str, score: float):
        # Convert score to success rate
        success_rate = max(0.0, min(1.0, score))

        # Get task index for learning progress tracking
        task_idx = list(self._curricula.keys()).index(id)

        # Collect data for learning progress
        self._lp_tracker.collect_data({f"tasks/{task_idx}": [success_rate]})

        # Update task weights based on learning progress
        lp_weights, _ = self._lp_tracker.calculate_dist()

        # Update weights based on learning progress
        for i, task_id in enumerate(self._curricula.keys()):
            if i < len(lp_weights):
                self._task_weights[task_id] = lp_weights[i]

        # Normalize weights
        total_weight = sum(self._task_weights.values())
        if total_weight > 0:
            self._task_weights = {k: v / total_weight for k, v in self._task_weights.items()}

        super().complete_task(id, score)
```

### Bidirectional Learning Progress Tracker
```python
class BidirectionalLearningProgress:
    def __init__(self, search_space, ema_timescale=0.001, progress_smoothing=0.05,
                 num_active_tasks=16, rand_task_rate=0.25, sample_threshold=10, memory=25):
        if isinstance(search_space, int):
            search_space = Discrete(search_space)
        
        self._search_space = search_space
        self._num_tasks = max_num_levels = search_space.n
        self._ema_timescale = ema_timescale
        self.progress_smoothing = progress_smoothing
        self.num_active_tasks = int(num_active_tasks)
        self._rand_task_rate = rand_task_rate
        self._sample_threshold = sample_threshold
        self._memory = int(memory)
        
        # Initialize outcome tracking for each task index
        self._outcomes = {}
        for i in range(max_num_levels):
            self._outcomes[i] = []
            
        # EMA tracking arrays
        self._p_fast = None
        self._p_slow = None
        self._p_true = None
        self._random_baseline = None
        
        # Task success rate tracking
        self._task_success_rate = np.zeros(max_num_levels)
        self._mean_samples_per_eval = []
        self._num_nans = []
        self._update_mask = np.ones(max_num_levels).astype(bool)
        self._sample_levels = np.arange(max_num_levels).astype(np.int32)
        self._counter = {i: 0 for i in self._sample_levels}
        self._task_dist = None
        self._stale_dist = True

    def collect_data(self, infos):
        """Collect task outcome data for learning progress tracking."""
        if not bool(infos):
            return

        for k, v in infos.items():
            if "tasks" in k:
                task_id = int(k.split("/")[1])
                for res in v:
                    self._outcomes[task_id].append(res)
                    if task_id in self._sample_levels:
                        self._counter[task_id] += 1

    def _update(self):
        """Update learning progress tracking with current task success rates."""
        task_success_rates = np.array([
            np.mean(self._outcomes[i]) if len(self._outcomes[i]) > 0 else DEFAULT_SUCCESS_RATE
            for i in range(self._num_tasks)
        ])
        
        # Handle NaN values
        task_success_rates = np.nan_to_num(task_success_rates, nan=DEFAULT_SUCCESS_RATE)

        if self._random_baseline is None:
            self._random_baseline = np.minimum(task_success_rates, RANDOM_BASELINE_CAP)

        # Normalize task success rates
        denominator = 1.0 - self._random_baseline[self._update_mask]
        denominator = np.where(denominator <= 0, 1.0, denominator)

        normalized_task_success_rates = (
            np.maximum(
                task_success_rates[self._update_mask] - self._random_baseline[self._update_mask],
                np.zeros(task_success_rates[self._update_mask].shape),
            ) / denominator
        )

        if self._p_fast is None:
            self._p_fast = normalized_task_success_rates[self._update_mask]
            self._p_slow = normalized_task_success_rates[self._update_mask]
            self._p_true = task_success_rates[self._update_mask]
        else:
            self._p_fast[self._update_mask] = (
                normalized_task_success_rates * self._ema_timescale +
                self._p_fast[self._update_mask] * (1.0 - self._ema_timescale)
            )
            self._p_slow[self._update_mask] = (
                self._p_fast[self._update_mask] * self._ema_timescale +
                self._p_slow[self._update_mask] * (1.0 - self._ema_timescale)
            )
            self._p_true[self._update_mask] = (
                task_success_rates[self._update_mask] * self._ema_timescale +
                self._p_true[self._update_mask] * (1.0 - self._ema_timescale)
            )

        self._stale_dist = True
        self._task_dist = None
        return task_success_rates

    def _learning_progress(self, reweight: bool = True) -> np.ndarray:
        """Calculate learning progress as difference between fast and slow EMAs."""
        fast = self._reweight(self._p_fast) if reweight else self._p_fast
        slow = self._reweight(self._p_slow) if reweight else self._p_slow
        return abs(fast - slow)

    def _reweight(self, probs: np.ndarray) -> np.ndarray:
        """Apply progress smoothing reweighting."""
        numerator = probs * (1.0 - self.progress_smoothing)
        denominator = probs + self.progress_smoothing * (1.0 - 2.0 * probs)
        denominator = np.where(denominator <= 0, 1.0, denominator)
        return numerator / denominator

    def _sample_distribution(self):
        """Generate task distribution based on learning progress."""
        task_dist = np.ones(self._num_tasks) / self._num_tasks
        learning_progress = self._learning_progress()

        posidxs = [i for i, lp in enumerate(learning_progress) if lp > 0 or self._p_true[i] > 0]
        any_progress = len(posidxs) > 0
        subprobs = learning_progress[posidxs] if any_progress else learning_progress

        std = np.std(subprobs)
        if std > 0:
            subprobs = (subprobs - np.mean(subprobs)) / std
        else:
            subprobs = subprobs - np.mean(subprobs)

        subprobs = self._sigmoid(subprobs)

        # Normalize to sum to 1
        sum_probs = np.sum(subprobs)
        if sum_probs > 0:
            subprobs = subprobs / sum_probs
        else:
            subprobs = np.ones_like(subprobs) / len(subprobs)

        if any_progress:
            task_dist = np.zeros(len(learning_progress))
            task_dist[posidxs] = subprobs
        else:
            task_dist = subprobs

        self._task_dist = task_dist.astype(np.float32)
        self._stale_dist = False

        # Memory management - trim old outcomes
        for i in range(self._num_tasks):
            self._outcomes[i] = self._outcomes[i][-self._memory:]
            
        return self._task_dist

    def calculate_dist(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate task distribution and sample levels."""
        if (all([v < self._sample_threshold for k, v in self._counter.items()]) and 
            self._random_baseline is not None):
            # Return cached results if under threshold
            if self._task_dist is None or len(self._task_dist) == 0:
                self._task_dist = np.ones(self._num_tasks) / self._num_tasks
            if self._sample_levels is None or len(self._sample_levels) == 0:
                self._sample_levels = np.arange(self._num_tasks).astype(np.int32)
            return self._task_dist, self._sample_levels

        self._task_success_rate = self._update()
        task_dist = self._sample_distribution()
        tasks = self._sample_tasks()
        return task_dist, tasks
```

**Characteristics:**
- **Complexity**: O(1) for task indexing, O(k) for weight calculation where k = fixed task space
- **Memory**: Bounded per task index with automatic trimming
- **Task Management**: Fixed task space with predefined indices (0 to n-1)
- **Sampling**: Sophisticated bidirectional learning progress with reweighting
- **Integration**: Minimal - only overrides `complete_task()` method

## Key Performance Advantages

### 1. Fixed Task Space (MettagGrid approach)
- **Task indexing**: `task_idx = list(self._curricula.keys()).index(id)` - O(1) operation
- **Predefined arrays**: All data structures sized at initialization
- **No dynamic allocation**: No task creation/destruction during training
- **Memory bounded**: `self._outcomes[i] = self._outcomes[i][-self._memory:]` per index

### 2. Minimal State Management
- **Single integration point**: Only `complete_task()` method modified
- **Inherited infrastructure**: Built on top of existing `RandomCurriculum`
- **Cached distributions**: Expensive calculations cached until task completion
- **Efficient sampling**: Uses existing curriculum sampling with updated weights

### 3. Bidirectional Learning Progress Algorithm
- **Fast/slow EMAs**: Dual exponential moving averages for progress detection
- **Progress smoothing**: Reweighting function to smooth learning progress signals
- **Random baseline**: Automatic baseline detection capped at 75%
- **Sigmoid normalization**: Converts learning progress to sampling probabilities

## Constants and Defaults

```python
# From mettagrid/learning_progress.py
DEFAULT_SUCCESS_RATE = 0.0
DEFAULT_WEIGHT = 1.0
RANDOM_BASELINE_CAP = 0.75

# Default parameters
ema_timescale = 0.001          # EMA learning rate
progress_smoothing = 0.05      # Progress reweighting factor
num_active_tasks = 16          # Number of active tasks to sample
rand_task_rate = 0.25          # Rate of random task selection
sample_threshold = 10          # Minimum samples before distribution update
memory = 25                    # Number of recent outcomes to remember per task
```

## Integration Pattern

The MettagGrid approach used a **hybrid inheritance pattern**:

1. **Inherit from RandomCurriculum**: Get all basic curriculum functionality
2. **Add learning progress tracker**: Initialize `BidirectionalLearningProgress` with task space size
3. **Override complete_task()**: Add learning progress tracking and weight updates
4. **Preserve existing API**: No changes to task sampling or generation logic

This minimal integration approach meant:
- **Low risk**: Only one method override, rest of functionality unchanged
- **High performance**: O(1) task indexing, cached weight calculations
- **Full compatibility**: Worked with existing curriculum infrastructure
- **Easy debugging**: Clear separation between base curriculum and learning progress

## File Locations at Commit `4167f6684`

```
metta/cogworks/curriculum/
├── learning_progress_minimal.py    # Simple individual task tracking
├── adapter.py                      # Curriculum adapters
├── test_minimal.py                 # Tests for minimal implementation
└── test_integration.py             # Integration tests

mettagrid/src/metta/mettagrid/curriculum/
├── learning_progress.py            # Sophisticated fixed-space tracking
├── bucketed.py                     # Bucketed curriculum (inherits from learning_progress)
└── random.py                       # Base RandomCurriculum
```

## State at Dehydration

By commit `028e96e2f` ("Dehydration"), both implementations were commented out with `# TODO #dehydration`, indicating they were disabled during the dehydration process rather than replaced with equivalent functionality.

This documentation captures the working state of learning progress curriculum before dehydration disabled it.