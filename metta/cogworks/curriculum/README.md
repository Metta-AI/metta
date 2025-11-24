# Curriculum Learning System

Adaptive task selection framework that automatically selects training tasks based on agent learning progress.

## Core Components

- **`Curriculum`**: Main curriculum manager coordinating task generation and selection
- **`LearningProgressAlgorithm`**: Intelligent task selection based on learning progress
- **`TaskTracker`**: Performance tracking with pluggable memory backends (Local/Shared)
- **`TaskGenerator`**: Flexible task generation (single, bucketed, or merged task sets)
- **`CurriculumEnv`**: PufferLib environment wrapper
- **`TaskState`**: Pydantic model for type-safe task data access

## Key Features

- **Bidirectional Learning Progress**: Tracks fast/slow EMAs to identify learning opportunities
- **Multi-Process Support**: Shared memory backend with atomic updates and cross-process synchronization
- **Pluggable Backends**: Local memory (single-process) or shared memory (multi-process)
- **Flexible Task Generation**: Single, bucketed, or merged task distributions
- **Comprehensive Statistics**: Per-task metrics, Gini coefficients, label tracking

## Quick Start

```python
from metta.cogworks.curriculum import LearningProgressConfig, Curriculum

# Use preset configurations
config = LearningProgressConfig.default(num_active_tasks=256)
# Also available: .stable() for noisy environments, .fast_learning() for fast learners

# Multi-process: use shared memory with session_id
config = LearningProgressConfig.default(
    num_active_tasks=256,
    session_id="my_training_session",  # All workers must use same ID
)

# Single-process: use local memory for better performance
config = LearningProgressConfig.default(num_active_tasks=256, use_shared_memory=False)

# Initialize and use
curriculum = Curriculum(config)
task = curriculum.get_task()
curriculum.update_task_performance(task._task_id, score=0.75)
```

## Configuration Presets

| Preset             | Use Case                      | EMA Convergence        | Exploration    |
| ------------------ | ----------------------------- | ---------------------- | -------------- |
| `.default()`       | Most RL environments          | Fast (~10 samples)     | Standard (0.1) |
| `.stable()`        | Noisy/stochastic environments | Slow (~100 samples)    | High (0.15)    |
| `.fast_learning()` | Fast-learning agents          | Very fast (~5 samples) | Low (0.05)     |

All presets support parameter overrides: `LearningProgressConfig.default(num_active_tasks=512, ema_timescale=0.05)`

## Architecture

```
TaskGenerator → Curriculum → LearningProgressAlgorithm → TaskTracker → Backend (Local/Shared)
                      ↓
                CurriculumEnv → Agent
```

### Memory Backends

| Backend                 | Use Case       | Storage              | Performance        |
| ----------------------- | -------------- | -------------------- | ------------------ |
| **LocalMemoryBackend**  | Single-process | Numpy arrays         | Fastest (no IPC)   |
| **SharedMemoryBackend** | Multi-process  | Shared memory + Lock | Cross-process sync |

**TaskState** provides type-safe access: `task_id`, `creation_time`, `is_active`, `num_completions`, `fast_ema`,
`slow_ema`, `success_threshold`, `seed`

**Operations**: O(1) for task updates, O(max_tasks) for cross-process aggregations

## Key Algorithms

**Learning Progress**: `lp_score = abs(fast_ema - slow_ema)` (bidirectional mode: normalized by random baseline)

**Task Sampling**: Calculate LP scores → add exploration bonus → z-score normalize → softmax → sample

**Task Eviction**: Evict tasks at capacity with minimum samples and low LP scores

## Recent Improvements

- **Pydantic Migration**: Type-safe `TaskState` models replace raw arrays with magic offsets
- **Pluggable Backends**: Unified `TaskTracker` with `create_task_tracker()` factory function
- **Configuration Presets**: `.default()`, `.stable()`, `.fast_learning()` helper methods
- **Performance**: O(1) fast path for updates, type-safe cold path for initialization
