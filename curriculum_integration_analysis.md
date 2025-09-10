# Curriculum Architecture Analysis: Current vs Learning Progress Integration

This document provides a comprehensive analysis of how curricula currently work in the Metta codebase and how the Learning Progress curriculum should integrate with the arena training pipeline.

## Current Curriculum Architecture

### 1. Current Codeflow for `./tools/run.py experiments.recipes.arena.train`

**Recipe Level (`experiments/recipes/arena.py:24-48`)**:
1. `make_curriculum()` creates a `BucketedTaskGeneratorConfig` via `cc.bucketed(arena_env)`
2. Adds bucket dimensions for inventory rewards, attack costs, object resources
3. Calls `arena_tasks.to_curriculum()` → returns `CurriculumConfig` with **NO algorithm_config** (defaults to `None`)

**Trainer Integration (`metta/rl/trainer.py:89`)**:
4. `Curriculum(trainer_cfg.curriculum)` instantiates the curriculum
5. **Critical**: Since `algorithm_config=None`, defaults to `DiscreteRandomConfig()` in `curriculum.py:377-378`
6. Creates a `DiscreteRandomCurriculum` algorithm that scores all tasks equally

**Current Default Algorithm**:
- **Type**: `DiscreteRandomCurriculum` (uniform random sampling)
- **Task Selection**: All tasks have equal probability (score = 1.0)
- **Eviction**: No preference - random eviction when pool is full
- **Performance Tracking**: Basic task tracker but no learning progress calculation

### 2. Current Task Lifecycle

```
Arena Recipe → BucketedTaskGeneratorConfig → CurriculumConfig(algorithm_config=None)
    ↓
Trainer → Curriculum(config) → DiscreteRandomCurriculum (default)
    ↓
Task Pool: 10,000 tasks, uniform sampling, random eviction
```

**Key Components**:
- `TaskTracker`: Tracks completion counts, scores, timing (curriculum.py:22-132)
- `BucketedTaskGenerator`: Generates env configs with different bucket parameter values
- `DiscreteRandomCurriculum`: Simple uniform sampling algorithm

## Learning Progress Curriculum Architecture

### 1. Learning Progress Components

The Learning Progress curriculum is **already implemented** and consists of:

**Core Algorithm (`learning_progress_algorithm.py:156-350`)**:
- `LearningProgressAlgorithm`: Main coordination class
- `LearningProgressConfig`: Hyperparameters (ema_timescale, exploration_bonus, etc.)

**Specialized Components**:
- `LearningProgressScorer`: Calculates variance-based learning progress scores
- `BucketAnalyzer` (in `stats.py`): Tracks completion patterns across bucket dimensions
- `TaskTracker`: Enhanced task memory and performance tracking (shared with base curriculum)

**Key Features**:
- **EMA Tracking**: Exponential moving averages of task performance and variance
- **Learning Progress Score**: High variance = actively learning, low variance = plateaued
- **Exploration Bonus**: New tasks get priority for exploration
- **Intelligent Eviction**: Evicts tasks in bottom 20% of learning progress scores
- **Bucket Analysis**: Tracks completion density across parameter space

### 2. Learning Progress Task Scoring

```python
# Core scoring logic in learning_progress_algorithm.py:46-79
def get_learning_progress_score(self, task_id: int, task_tracker: TaskTracker) -> float:
    if completion_count < 2:
        return self.exploration_bonus  # New tasks get exploration priority
    
    # Calculate variance from EMA tracking
    variance = max(0.0, ema_squared - ema_score * ema_score)
    learning_progress = sqrt(variance)  # High variance = active learning
    
    # Boost tasks with few samples
    if num_samples < 10:
        learning_progress += exploration_bonus * (10 - num_samples) / 10
    
    return learning_progress
```

## Integration Analysis: What Currently Works vs What Needs Changes

### ✅ What Already Works

**Navigation Recipe Integration**:
- `navigation.py:95-104` **already uses Learning Progress curriculum**!
- Explicitly sets `algorithm_config=LearningProgressConfig()` 
- Has optimized parameters: `ema_timescale=0.001`, `exploration_bonus=0.1`
- Works with 1,000 active tasks and proper bucket analysis

**Architecture Compatibility**:
- `CurriculumConfig` supports `algorithm_config` field (curriculum.py:329-331)
- `LearningProgressConfig.create()` properly instantiates `LearningProgressAlgorithm`
- All task lifecycle hooks are implemented (`on_task_created`, `on_task_evicted`, `update_task_performance`)

### ❌ Arena Recipe Gap

**Missing Learning Progress Integration**:
```python
# Current: experiments/recipes/arena.py:48
return arena_tasks.to_curriculum()  # algorithm_config=None → defaults to DiscreteRandom

# Should be:
return CurriculumConfig(
    task_generator=arena_tasks,
    algorithm_config=LearningProgressConfig(
        ema_timescale=0.001,
        exploration_bonus=0.1,
        max_memory_tasks=1000,
        enable_detailed_bucket_logging=False,  # performance
    ),
)
```

## Required Changes for Arena Integration

### 1. Minimal Change: Update Arena Recipe

**File**: `experiments/recipes/arena.py:24-48`

```python
def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    use_learning_progress: bool = True,  # Add parameter
) -> CurriculumConfig:
    arena_env = arena_env or make_mettagrid()
    arena_tasks = cc.bucketed(arena_env)
    
    # ... existing bucket configuration ...
    
    if use_learning_progress:
        return CurriculumConfig(
            task_generator=arena_tasks,
            algorithm_config=LearningProgressConfig(
                ema_timescale=0.001,
                exploration_bonus=0.1,
                max_memory_tasks=1000,
                enable_detailed_bucket_logging=False,
            ),
        )
    else:
        return arena_tasks.to_curriculum()  # Keep backward compatibility
```

### 2. Update Arena Train Function

```python
def train(
    curriculum: Optional[CurriculumConfig] = None,
    use_learning_progress: bool = True,  # Add parameter
) -> TrainTool:
    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
        curriculum=curriculum or make_curriculum(use_learning_progress=use_learning_progress),
        # ... rest unchanged
    )
    return TrainTool(trainer=trainer_cfg)
```

## Performance and Logging Impact

### 1. Learning Progress Overhead

**Computational Cost**:
- EMA updates: O(1) per task completion 
- Score calculation: O(k) where k = active tasks (~10,000)
- Bucket analysis: Configurable via `enable_detailed_bucket_logging`

**Memory Usage**:
- Task tracker: ~1,000 tasks max (configurable)
- EMA data: 3 floats per task = ~12KB for 1,000 tasks
- Bucket analysis: Depends on bucket dimensions (3 max by default)

### 2. Enhanced Logging

**New Wandb Metrics** (with `algorithm/` prefix):
- `algorithm/tracker/total_completions`: Task completion counts
- `algorithm/lp/mean_learning_progress`: Average learning progress across tasks
- `algorithm/buckets/completion_entropy`: Exploration coverage across parameter space

**Detailed Bucket Logging** (expensive, disabled by default):
- Per-bucket completion density statistics
- Parameter space coverage analysis

## Implementation Priority

### Phase 1: Basic Integration (Recommended)
1. Add `use_learning_progress` parameter to arena recipe
2. Match navigation recipe's Learning Progress configuration
3. Test with existing arena training pipeline

### Phase 2: Configuration Refinement (Optional)
1. Tune Learning Progress hyperparameters for arena environment
2. Enable selective detailed bucket logging for specific experiments
3. A/B test Learning Progress vs DiscreteRandom performance

## Code Quality Considerations

Based on Jack's feedback, these architectural improvements should be made:

### 1. Component Organization Issues
- **TaskTracker**: Currently in `curriculum.py:22-132` but should be curriculum-general, not LP-specific
- **BucketAnalyzer**: Renamed from `bucket_analyzer.py` to `stats.py` but should be further generalized
- **LearningProgressScorer**: Currently separate file but core algorithm logic should be in main algorithm file

### 2. Configuration Defaults
- Navigation recipe hardcodes LP config (lines 98-103) - should move to component-level defaults
- Arena recipe should use same pattern for consistency

### 3. Code Quality Issues
```python
# Current problematic patterns to fix:
def __getattribute__(self, name: str):  # curriculum_env.py - avoid this pattern
hasattr(self, "_first_reset_done")      # curriculum_env.py - set in __init__ instead
from .curriculum import CurriculumConfig # task_generator.py - avoid inline imports
```

## Summary

The **Learning Progress curriculum is fully implemented and working** in the navigation recipe. The arena recipe simply needs a **one-line change** to enable it:

```python
# Change this:
return arena_tasks.to_curriculum()

# To this:  
return CurriculumConfig(task_generator=arena_tasks, algorithm_config=LearningProgressConfig())
```

The navigation recipe serves as the **reference implementation** showing how Learning Progress curriculum should be integrated. The arena recipe currently defaults to uniform random sampling, but enabling Learning Progress should provide more intelligent task selection based on learning difficulty and exploration needs.