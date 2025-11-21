# Curriculum Learning System

Comprehensive curriculum learning framework for adaptive task selection and intelligent training. The system
automatically selects training tasks based on agent learning progress, enabling more efficient and effective training on
complex task distributions.

## Core Components

### Main Classes

- **`Curriculum`**: Main curriculum manager coordinating task generation and selection
- **`LearningProgressAlgorithm`**: Intelligent task selection based on learning progress
- **`TaskTracker`**: Performance tracking with shared memory support for multi-process training
- **`TaskGenerator`**: Flexible task generation (single, bucketed, or merged task sets)

### Supporting Components

- **`CurriculumEnv`**: Wrapper environment that integrates curriculum with PufferLib
- **`SharedMemoryBackend`**: Cross-process data sharing with synchronization
- **`LPScorer`**: Learning progress scoring strategies (Variance and bidirectional)

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

Future Features will add a exploration-exploitation pool design (removed to reduce diff)
