# Curriculum System Architecture

## Overview

The curriculum system provides intelligent task selection and generation for reinforcement learning training. It separates concerns into distinct layers: task generation, task management, and learning progress algorithms.

## Architecture Diagram

```mermaid
graph TB
    subgraph "Core Curriculum Layer"
        Curriculum[Curriculum<br/>Task pool manager]
        CurriculumTask[CurriculumTask<br/>Task instance + metadata]
        CurriculumConfig[CurriculumConfig<br/>Configuration]
    end

    subgraph "Algorithm Layer"
        CurriculumAlgorithm[CurriculumAlgorithm<br/>Abstract scoring interface]
        DiscreteRandom[DiscreteRandomCurriculum<br/>Uniform selection]
        LearningProgress[LearningProgressAlgorithm<br/>LP-based selection]
    end

    subgraph "Task Generation Layer"
        TaskGenerator[TaskGenerator<br/>Deterministic generation]
        SingleTask[SingleTaskGenerator<br/>Fixed task]
        TaskSet[TaskGeneratorSet<br/>Weighted sampling]
        Bucketed[BucketedTaskGenerator<br/>Parameterized tasks]
    end

    subgraph "Learning Progress Components"
        TaskTracker[TaskTracker<br/>Unified tracking]
        LPScorer[LPScorer<br/>Scoring strategy]
        Bidirectional[BidirectionalLPScorer<br/>Fast/slow EMA]
        Basic[BasicLPScorer<br/>Variance-based]
        MemoryBackend[TaskMemoryBackend<br/>Storage interface]
        LocalMemory[LocalMemoryBackend<br/>Single-process]
        SharedMemory[SharedMemoryBackend<br/>Multi-process]
    end

    subgraph "Support Systems"
        StatsLogger[StatsLogger<br/>Statistics interface]
        SliceAnalyzer[SliceAnalyzer<br/>Slice analysis]
        StatsAggregator[LPStatsAggregator<br/>Stats computation]
        CacheCoordinator[CacheCoordinator<br/>Cache management]
    end

    %% Core relationships
    Curriculum --> CurriculumConfig
    Curriculum --> CurriculumTask
    Curriculum --> CurriculumAlgorithm
    Curriculum --> TaskGenerator

    %% Algorithm inheritance
    CurriculumAlgorithm <|-- DiscreteRandom
    CurriculumAlgorithm <|-- LearningProgress

    %% Task generation inheritance
    TaskGenerator <|-- SingleTask
    TaskGenerator <|-- TaskSet
    TaskGenerator <|-- Bucketed
    TaskSet --> TaskGenerator: contains multiple
    Bucketed --> TaskGenerator: wraps child

    %% Learning progress dependencies
    LearningProgress --> TaskTracker
    LearningProgress --> LPScorer
    LearningProgress --> StatsAggregator
    LearningProgress --> CacheCoordinator

    %% Scorer strategy
    LPScorer <|-- Bidirectional
    LPScorer <|-- Basic
    LearningProgress --> LPScorer: uses

    %% Backend strategy
    TaskTracker --> MemoryBackend
    MemoryBackend <|-- LocalMemory
    MemoryBackend <|-- SharedMemory

    %% Stats inheritance
    StatsLogger <|-- CurriculumAlgorithm
    StatsLogger <|-- Curriculum
    CurriculumAlgorithm --> SliceAnalyzer
    StatsAggregator --> TaskTracker
    StatsAggregator --> LPScorer
    StatsAggregator --> SliceAnalyzer

    %% Styling
    classDef coreClass fill:#e1f5ff,stroke:#01579b
    classDef algoClass fill:#f3e5f5,stroke:#4a148c
    classDef genClass fill:#e8f5e9,stroke:#1b5e20
    classDef lpClass fill:#fff3e0,stroke:#e65100
    classDef supportClass fill:#fce4ec,stroke:#880e4f

    class Curriculum,CurriculumTask,CurriculumConfig coreClass
    class CurriculumAlgorithm,DiscreteRandom,LearningProgress algoClass
    class TaskGenerator,SingleTask,TaskSet,Bucketed genClass
    class TaskTracker,LPScorer,Bidirectional,Basic,MemoryBackend,LocalMemory,SharedMemory lpClass
    class StatsLogger,SliceAnalyzer,StatsAggregator,CacheCoordinator supportClass
```

## Component Descriptions

### Core Curriculum Layer

**Curriculum** - Main orchestrator that:
- Maintains active task pool (configurable size)
- Delegates task generation to `TaskGenerator`
- Delegates task selection to `CurriculumAlgorithm`
- Manages task lifecycle (creation, eviction)
- Provides unified statistics interface via `StatsLogger`

**CurriculumTask** - Individual task instance containing:
- Task ID (deterministic seed)
- Environment configuration (`MettaGridConfig`)
- Performance metadata (completions, scores)
- Slice/bucket values for analysis

**CurriculumConfig** - Configuration specifying:
- Task generator configuration
- Algorithm configuration (optional)
- Pool size and eviction parameters
- Random seed for reproducibility

### Algorithm Layer

**CurriculumAlgorithm** - Abstract interface for task selection:
- `score_tasks()`: Assigns scores to tasks for selection
- `recommend_eviction()`: Suggests which task to evict
- `should_evict_task()`: Checks if task meets eviction criteria
- `update_task_performance()`: Receives task completion feedback
- Inherits from `StatsLogger` for unified statistics

**DiscreteRandomCurriculum** - Simplest algorithm:
- All tasks have equal probability
- Random eviction selection
- No performance tracking needed

**LearningProgressAlgorithm** - Intelligent algorithm:
- Tracks task performance via `TaskTracker`
- Scores tasks using `LPScorer` strategy
- Evicts low-progress tasks
- Provides detailed statistics

### Task Generation Layer

**TaskGenerator** - Base class for deterministic generation:
- `get_task(task_id)`: Generate `MettaGridConfig` from task ID
- Uses task_id as random seed for reproducibility
- Validates environment invariants (resources, actions, agents)
- Supports config overrides

**SingleTaskGenerator** - Always returns same task:
- Wraps fixed `MettaGridConfig`
- Useful for single-task training

**TaskGeneratorSet** - Samples from multiple generators:
- Weighted sampling from child generators
- Each sample re-seeds based on task_id

**BucketedTaskGenerator** - Parameterized generation:
- Defines buckets of parameter values
- Samples from buckets deterministically
- Applies as overrides to child generator
- Tracks slice values for analysis

### Learning Progress Components

**TaskTracker** - Unified task tracking:
- Configurable memory backend (local vs shared)
- Tracks: completions, scores, EMAs, LP scores
- Thread-safe operations with locking
- Checkpointing support

**LPScorer** - Strategy pattern for scoring:
- Abstract interface for different LP algorithms
- Separates scoring logic from tracking
- Supports caching for efficiency

**BidirectionalLPScorer** (default):
- Fast and slow EMAs to detect learning progress
- Exploration bonus for rarely-seen tasks
- Performance bonus (configurable)
- Reweighting based on overall progress

**BasicLPScorer** (legacy):
- Variance-based learning progress
- Adaptive alpha for EMA
- Simpler but less sophisticated

**TaskMemoryBackend** - Storage abstraction:
- Unified interface for task data
- Enables identical code for single/multi-process

**LocalMemoryBackend**:
- NumPy arrays in process memory
- No IPC overhead
- Threading-level locks

**SharedMemoryBackend**:
- `multiprocessing.shared_memory`
- Cross-process data sharing
- Process-safe locks
- Session-based namespacing

### Support Systems

**StatsLogger** - Unified statistics interface:
- Base stats (always computed)
- Detailed stats (expensive, optional)
- Caching with invalidation
- Prefix support for namespacing

**SliceAnalyzer** - Performance analysis by slices:
- Tracks task completion by slice values
- Computes mean/variance per slice
- Identifies difficult parameter regions
- Configurable max axes (performance tuning)

**LPStatsAggregator** - Centralizes stats computation:
- Aggregates from tracker, scorer, analyzer
- Avoids duplicate computation
- Consistent stat formatting

**CacheCoordinator** - Cache invalidation:
- Centralizes invalidation logic
- Invalidates across stats, scorer, analyzer
- Triggered by task state changes

## Design Principles

### 1. Separation of Concerns
- **Task generation** is independent of **task selection**
- **Storage backend** is abstracted from **tracking logic**
- **Scoring strategy** is decoupled from **algorithm**

### 2. Strategy Pattern
- `CurriculumAlgorithm` for selection strategies
- `LPScorer` for scoring algorithms
- `TaskMemoryBackend` for storage implementations

### 3. Unified Interfaces
- `StatsLogger` provides consistent statistics across components
- `TaskMemoryBackend` allows identical code for local/shared memory
- All algorithms use same `score_tasks()` interface

### 4. Configurability
- Pydantic configs for all components
- Hierarchical configuration (Curriculum → Algorithm → Scorer)
- CLI overrides supported at all levels

### 5. Extensibility
- New algorithms extend `CurriculumAlgorithm`
- New scorers extend `LPScorer`
- New generators extend `TaskGenerator`
- New backends extend `TaskMemoryBackend`

## Data Flow

### Task Selection Flow
```
1. Curriculum.get_task()  ← Called independently by each environment
2. → Algorithm.score_tasks(active_tasks)
3.   → Scorer.score_task(task_id)
4.     → TaskTracker.get_task_stats(task_id)
5.       → MemoryBackend.get_task_data(index)
6. → Weighted sampling from scored tasks (WITH REPLACEMENT)
7. → Return CurriculumTask
    └─> Task remains in pool (not consumed)
    └─> Multiple environments can receive same task_id
```

**Note:** Tasks are sampled **with replacement**. In vectorized training:
- Multiple environments can run the same task_id simultaneously
- High-LP tasks naturally appear in more environments
- Tasks only leave the pool via eviction, not selection

### Task Completion Flow
```
1. Curriculum.update_task_performance(task_id, score)
2. → Algorithm.update_task_performance(task_id, score)
3.   → Scorer.update_with_score(task_id, score)
4.   → Scorer.score_task(task_id) to get LP score
5.   → TaskTracker.update_task_performance(task_id, score, lp_score)
6.     → MemoryBackend.get_task_data(index) - read current
7.     → Update EMAs, counts, totals
8.     → MemoryBackend.get_task_data(index) - write updated
9.   → CacheCoordinator.invalidate_stats_cache()
```

### Task Eviction Flow
```
1. Curriculum: Pool at capacity, check eviction
2. → Algorithm.should_evict_task(task_id)
3.   → Check completion count >= min_presentations
4.   → Compare LP score to percentile threshold
5. → If evictable: Algorithm.recommend_eviction(task_ids)
6.   → Return task with minimum LP score
7. → Curriculum._evict_specific_task(task_id)
8.   → Algorithm.on_task_evicted(task_id)
9.     → Scorer.remove_task(task_id)
10.    → TaskTracker.remove_task(task_id)
```

## Key Refactor Changes

### Before Refactor
- Task tracking mixed with algorithm logic
- Conditional code for centralized vs local tracking
- Hardcoded to learning progress algorithm
- Limited extensibility

### After Refactor
- **Clear separation**: TaskTracker handles tracking, Algorithm handles selection
- **Strategy pattern**: Swap algorithms and scorers via config
- **Backend abstraction**: Identical code for local/shared memory
- **Stats unification**: StatsLogger provides consistent interface
- **Improved testability**: Each component tests independently

### Removed Parameters
- `max_memory_tasks`: Now derived from `num_active_tasks` in `LearningProgressConfig`
- Algorithm-specific params moved to respective config classes

### New Capabilities
- Multiple scoring strategies without changing core code
- New algorithms without modifying Curriculum
- New storage backends without changing TaskTracker
- Comprehensive statistics with caching

## File Organization

```
metta/cogworks/curriculum/
├── curriculum.py              # Core: Curriculum, CurriculumTask, CurriculumAlgorithm
├── task_generator.py          # Task generation: TaskGenerator, SingleTask, TaskSet, Bucketed
├── learning_progress_algorithm.py  # LearningProgressAlgorithm + config
├── task_tracker.py            # TaskTracker (unified implementation)
├── lp_scorers.py              # LPScorer strategies (Bidirectional, Basic)
├── shared_memory_backend.py   # TaskMemoryBackend implementations (Local, Shared)
├── stats.py                   # StatsLogger, SliceAnalyzer, StatsAggregator, CacheCoordinator
├── demo.py                    # Example usage patterns
└── structure.md               # This file
```

## Usage Examples

### Simple Random Curriculum
```python
config = CurriculumConfig(
    task_generator=SingleTaskGenerator.Config(env=my_env_config),
    num_active_tasks=100,
    algorithm_config=None,  # Uses DiscreteRandomCurriculum
)
curriculum = config.make()
```

### Learning Progress with Local Memory
```python
config = CurriculumConfig(
    task_generator=BucketedTaskGenerator.Config(
        child_generator_config=SingleTaskGenerator.Config(env=base_config),
        buckets={"game.num_plants": [5, 10, 15, 20]},
    ),
    algorithm_config=LearningProgressConfig(
        num_active_tasks=1000,
        use_shared_memory=False,  # Local memory for single-process
        use_bidirectional=True,
    ),
)
curriculum = config.make()
```

### Learning Progress with Shared Memory
```python
config = CurriculumConfig(
    task_generator=my_task_generator_config,
    algorithm_config=LearningProgressConfig(
        num_active_tasks=10000,
        use_shared_memory=True,
        session_id="my_training_run",  # All workers share this
        use_bidirectional=True,
    ),
)
curriculum = config.make()
# All workers with same session_id share task tracking
```

### Custom Task Generator
```python
class MyTaskGenerator(TaskGenerator):
    class Config(TaskGeneratorConfig["MyTaskGenerator"]):
        difficulty_range: tuple[float, float] = (0.0, 1.0)

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        difficulty = rng.uniform(*self._config.difficulty_range)
        # Generate task based on difficulty
        return my_env_config
```

## Testing

Each component has dedicated tests:
- `test_curriculum.py`: Core curriculum logic
- `test_task_generator.py`: Task generation
- `test_learning_progress_algorithm.py`: LP algorithm
- `test_task_tracker.py`: Task tracking
- `test_shared_memory_backend.py`: Memory backends
- `test_stats.py`: Statistics systems

Mock implementations enable isolated testing of each layer.

