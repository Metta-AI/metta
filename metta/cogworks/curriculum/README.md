# Curriculum Learning System

Comprehensive curriculum learning framework for adaptive task selection and intelligent training. The system automatically selects training tasks based on agent learning progress, enabling more efficient and effective training on complex task distributions.

## Core Components

### Main Classes

- **`Curriculum`**: Main curriculum manager coordinating task generation and selection
- **`LearningProgressAlgorithm`**: Intelligent task selection based on learning progress
- **`TaskTracker`**: Performance tracking with shared memory support for multi-process training
- **`TaskGenerator`**: Flexible task generation (single, bucketed, or merged task sets)

### Supporting Components

- **`CurriculumEnv`**: Wrapper environment that integrates curriculum with PufferLib
- **`SharedMemoryBackend`**: Cross-process data sharing with proper synchronization
- **`LPScorer`**: Learning progress scoring strategies (standard and bidirectional)

## Key Features

### Bidirectional Learning Progress
Tracks fast/slow EMAs to detect learning opportunities:
- Fast EMA responds quickly to recent performance changes
- Slow EMA provides stable baseline
- Learning progress = |fast - slow| indicates learning potential

### Shared Memory Backend
True multi-process training with atomic updates:
- Manager.Lock() for proper cross-process synchronization
- Deterministic SHA256-based label hashing
- Clean slot management prevents data corruption
- Cross-process task visibility for accurate statistics

### Comprehensive Statistics
Per-task metrics, Gini coefficients, and learning progress tracking:
- Per-task: completion counts, success rates, LP scores
- Per-label: aggregated completions, eviction counts
- Gini coefficients: measure task distribution uniformity
- Epoch-level tracking: evictions, sampling counts

### Flexible Task Generation
Support for parameterized, bucketed, and merged task distributions:
- **SingleTaskGenerator**: Single fixed environment configuration
- **BucketedTaskGenerator**: Tasks organized into difficulty buckets
- **TaskGeneratorSet**: Merge multiple generators with weights

## Quick Start

### Basic Curriculum with Learning Progress

```python
from metta.cogworks.curriculum import LearningProgressConfig, Curriculum

# Create configuration
config = LearningProgressConfig(
    use_bidirectional=True,
    num_active_tasks=256,
    use_shared_memory=True,
)

# Initialize curriculum
curriculum = Curriculum(config)

# Get a task
task = curriculum.get_task()
env_cfg = task.get_env_cfg()

# Update performance after completion
curriculum.update_task_performance(task._task_id, score=0.75)
```

### Multi-Process Training

```python
from metta.cogworks.curriculum import LearningProgressConfig

# Shared memory configuration (same session_id across all processes)
config = LearningProgressConfig(
    use_bidirectional=True,
    num_active_tasks=256,
    use_shared_memory=True,
    session_id="my_training_session",  # All processes must use same ID
)

# In each worker process
curriculum = Curriculum(config)
# Workers automatically connect to shared memory
```

## Architecture

### Data Flow

```
TaskGenerator → Curriculum → CurriculumEnv → Agent
                    ↓
        LearningProgressAlgorithm
                    ↓
              TaskTracker
                    ↓
          SharedMemoryBackend
```

### Multi-Process Design

Each worker process has:
- **Local**: `_task_id_to_index` mapping for O(1) lookups
- **Local**: `_label_hash_to_string` mapping for label resolution
- **Shared**: Task performance data in shared memory
- **Shared**: Manager.Lock() for synchronization

Operations:
- **Fast path (O(1))**: Task updates, queries by task_id
- **Slow path (O(max_tasks))**: Cross-process aggregations, label statistics

## Key Algorithms

### Learning Progress Calculation

```python
# Standard LP: Rate of change in performance
lp_score = abs(fast_ema - slow_ema)

# Bidirectional LP: Compare against random baseline
p_fast = fast_ema - random_baseline
p_slow = slow_ema - random_baseline
lp_score = abs(p_fast - p_slow)
```

### Task Sampling

1. Calculate LP scores for all tasks
2. Add exploration bonus for under-sampled tasks
3. Z-score normalization
4. Softmax over scores → sampling probabilities
5. Sample task proportional to probability

### Task Eviction

Tasks are evicted when:
- Pool is at capacity AND
- Task has minimum presentations AND
- Task has low learning progress (no longer useful)

## Configuration

### LearningProgressConfig

Key parameters:

```python
LearningProgressConfig(
    # Pool size
    num_active_tasks=256,
    
    # Learning progress
    use_bidirectional=True,
    ema_timescale=0.1,
    min_samples_for_lp=5,
    
    # Task eviction
    eviction_mode="lp",
    max_stale_epochs=10,
    
    # Shared memory
    use_shared_memory=True,
    session_id="training_001",
    
    # Sampling
    temperature=1.0,
    exploration_bonus=0.1,
)
```

## Statistics and Monitoring

### Available Metrics

```python
# Get curriculum statistics
stats = curriculum.get_curriculum_stats()

# Contains:
# - mean_task_success_rate
# - mean_lp_score  
# - num_zeros_lp_dist
# - per_label_completions
# - per_label_evictions_this_epoch
# - gini coefficients
```

### Gini Coefficients

Measure task distribution uniformity (0 = perfectly uniform, 1 = all on one task):
- `pool_occupancy`: Based on completion counts
- `raw_lp_scores`: Based on learning progress values
- `z_scored_lp`: After z-score normalization
- `sampling_probs`: Final sampling distribution

## Recent Fixes (November 2025)

Six critical issues fixed:

1. **Lock Synchronization**: Now uses Manager().Lock() for proper cross-process sync
2. **Deterministic Hashing**: SHA256-based label hashing consistent across processes
3. **Stale Label Hashes**: Cleared on task removal to prevent corruption
4. **Cross-Process Visibility**: Scan all slots for accurate label statistics
5. **Eviction Counters**: Accumulate properly across episodes in an epoch
6. **Zero-Count Metric**: Fixed list comparison for accurate LP metrics

See `outputs/` directory for detailed fix documentation.

## File Organization

```
metta/cogworks/curriculum/
├── README.md                          # This file
├── __init__.py                        # Package exports
├── curriculum.py                      # Main Curriculum class
├── curriculum_base.py                 # Abstract base classes
├── curriculum_env.py                  # PufferLib environment wrapper
├── learning_progress_algorithm.py    # LP algorithm implementation
├── task_tracker.py                    # Performance tracking
├── task_generator.py                  # Task generation strategies
├── lp_scorers.py                      # LP scoring strategies
├── shared_memory_backend.py           # Memory backend implementations
└── stats.py                           # Statistics utilities
```

## Testing

Run curriculum tests:

```bash
uv run pytest tests/cogworks/curriculum/ -v
```

Key test files:
- `test_curriculum_core.py`: Basic functionality
- `test_curriculum_algorithms.py`: LP algorithm behavior
- `test_curriculum_shared_memory.py`: Multi-process coordination
- `test_curriculum_checkpointing.py`: State persistence

## Best Practices

### DO

✅ Use shared memory for multi-process training
✅ Set consistent `session_id` across all workers  
✅ Call `reset_epoch_counters()` at epoch boundaries
✅ Use `get_evictions_this_epoch()` for per-episode reporting
✅ Let training loop call `get_and_reset_evictions_this_epoch()` at epoch end

### DON'T

❌ Call `get_and_reset_evictions_this_epoch()` per-episode
❌ Mix `use_shared_memory=True` and `False` in same training run
❌ Forget to call `cleanup_shared_memory()` on shutdown
❌ Change `session_id` between checkpoint save and load

## Troubleshooting

### Shared Memory Issues

**Problem**: "Shared memory already exists" error
```bash
# Clean up stale shared memory (Linux)
ls /dev/shm/ta_* && rm /dev/shm/ta_*

# Or set a unique session_id
config.session_id = f"training_{uuid.uuid4().hex[:6]}"
```

**Problem**: Tasks from other processes not visible
- ✅ Fixed in recent updates - uses slot scanning

**Problem**: Incorrect label statistics
- ✅ Fixed in recent updates - deterministic hashing

### Performance Issues

**Problem**: Slow task updates
- Check if using O(1) task_id lookups (should be fast)
- Avoid calling expensive operations on hot path

**Problem**: High memory usage
- Reduce `num_active_tasks`
- Use smaller `task_struct_size` if possible

## See Also

- `learning_progress_algorithm.py`: Core LP algorithm with bidirectional scoring
- `task_tracker.py`: Performance tracking with shared memory support
- `curriculum.py`: Main Curriculum class coordinating all components
- `docs/`: Additional documentation and design notes
- `outputs/`: Fix documentation and implementation notes

## References

- Graves et al. (2017): Automated Curriculum Learning for Neural Networks
- Portelas et al. (2020): Teacher algorithms for curriculum learning of Deep RL
- OpenAI (2019): Emergent Tool Use from Multi-Agent Interaction

