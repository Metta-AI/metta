# Implementation Plan: Direct Integration with SliceAnalyzer

## Overview

This plan integrates the `learning_progress_modules` content directly into the main curriculum folder structure, renames BucketAnalyzer to SliceAnalyzer for better terminology, makes bidirectional learning progress the default, and creates a unified stats system.

## Key Changes Summary

### Terminology Updates
- **BucketAnalyzer** → **SliceAnalyzer**
- **bucket_analyzer** → **slice_analyzer**
- **enable_detailed_bucket_logging** → **enable_detailed_slice_logging**
- **max_bucket_axes** → **max_slice_axes**
- **bucket_values** → **slice_values**

### Architecture Changes
- Remove `learning_progress_modules/` folder entirely
- Integrate components directly into main curriculum folder
- Make bidirectional learning progress the default scoring method
- Create unified stats system with `StatsLogger` base class

## Module Integration Mapping

```
learning_progress_modules/                    →  Target Integration Location
├── task_tracker.py                          →  metta/cogworks/curriculum/task_tracker.py (standalone file)
├── bucket_analyzer.py                       →  metta/cogworks/curriculum/stats.py (integrated as SliceAnalyzer class)
├── bidirectional_learning_progress_scorer.py →  metta/cogworks/curriculum/learning_progress_algorithm.py (integrated as scoring method)
├── learning_progress_scorer.py              →  [DEPRECATED - replaced by bidirectional approach]
└── __init__.py                              →  [REMOVED - no longer needed]
```

## Final File Structure

```
metta/cogworks/curriculum/
├── __init__.py                         [UPDATE - Import SliceAnalyzer, not BucketAnalyzer]
├── curriculum.py                       [UPDATE - Use slice_analyzer, not bucket_analyzer]
├── curriculum_env.py                   [MINOR UPDATE - Batched stats]
├── task_generator.py                   [UPDATE - Add to_curriculum() method]
├── task_tracker.py                     [NEW - Direct from learning_progress_modules/]
├── stats.py                           [NEW - Contains StatsLogger + SliceAnalyzer]
├── learning_progress_algorithm.py      [MAJOR UPDATE - Integrate bidirectional + use slice_analyzer]
└── demo.py                            [MINOR UPDATE]

REMOVED:
├── learning_progress_modules/          [ENTIRE FOLDER DELETED]
├── learning_progress.py               [DELETED - Replaced by new algorithm]
└── task.py                            [DELETED - Replaced by CurriculumTask]
```

## Implementation Steps

### Phase 1: Foundation Setup

#### Step 1.1: Create stats.py (Unified Stats + SliceAnalyzer)
- Create `StatsLogger` abstract base class for all curriculum components
- Integrate `SliceAnalyzer` class (moved from bucket_analyzer.py with renamed terminology)
- Provide unified caching, prefixing, and detailed logging controls

#### Step 1.2: Move task_tracker.py
- Move `learning_progress_modules/task_tracker.py` → `curriculum/task_tracker.py`
- Update import paths from relative to absolute

### Phase 2: Core System Updates

#### Step 2.1: Update curriculum.py (SliceAnalyzer Integration)
- Make `CurriculumAlgorithm` inherit from `StatsLogger`
- Make `Curriculum` inherit from `StatsLogger`
- All algorithms get slice analysis capability by default
- Standardized stats interface across all algorithms

#### Step 2.2: Major Update learning_progress_algorithm.py (Bidirectional Integration)
- Integrate bidirectional scoring as default implementation
- Remove dependency on separate `LearningProgressScorer` class
- Add `use_bidirectional: bool = True` to `LearningProgressConfig`
- Integrate with unified stats system from `stats.py`

### Phase 3: Update Imports and Cleanup

#### Step 3.1: Update __init__.py (Remove Modules Folder)
- Remove imports from `learning_progress_modules`
- Add direct imports for `TaskTracker`, `SliceAnalyzer`, etc.
- Update `__all__` exports

#### Step 3.2: Delete learning_progress_modules Folder
- Remove entire `learning_progress_modules/` directory

#### Step 3.3: Update Recipes (Bidirectional by Default)
- Update `navigation.py` to use bidirectional by default
- Update `arena.py` to use bidirectional by default
- Use new slice terminology in parameter names

## Information Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NAVIGATION TRAINING FLOW (SliceAnalyzer)             │
└─────────────────────────────────────────────────────────────────────────┘

1. recipes/navigation.py::train()
   ├── Creates TrainerConfig with curriculum=make_curriculum()
   └── make_curriculum() returns CurriculumConfig with:
       ├── task_generator: BucketedTaskGeneratorConfig (dense + sparse tasks)
       └── algorithm_config: LearningProgressConfig(use_bidirectional=True)

2. metta/rl/trainer.py::train()
   ├── Creates CurriculumEnv(env, Curriculum(config))
   └── Runs training loop calling env.step() and env.reset()

3. CurriculumEnv workflow:
   ├── reset() → curriculum.get_task() → LearningProgressAlgorithm.score_tasks()
   ├── step() → collects episode rewards
   └── episode_end() → curriculum.update_task_performance()

4. LearningProgressAlgorithm (bidirectional):
   ├── score_tasks() → Integrated bidirectional scoring methods
   ├── update_task_performance() → TaskTracker.update_task() + slice_analyzer.update_task_completion()
   └── stats() → TaskTracker.get_global_stats() + SliceAnalyzer.get_slice_distribution_stats()

5. stats.py::StatsLogger (inherited by all components):
   ├── get_base_stats() → Required stats from all algorithms
   ├── get_detailed_stats() → Expensive slice analysis (if enabled)
   └── stats() → Cached, prefixed, unified logging output

DATA FLOW:
nav.py → TrainerConfig → CurriculumEnv → Curriculum → LearningProgressAlgorithm
                                              ↓
episode_rewards → update_task_performance → TaskTracker + SliceAnalyzer + BidirectionalScorer
                                              ↓
                                    stats.py → unified logging → W&B
```

## Implementation Priority & Dependencies

### Critical Path (Must be done in order):
1. **Create `stats.py`** → Foundation with `StatsLogger` + integrated `SliceAnalyzer`
2. **Move `task_tracker.py`** → Direct move from modules to curriculum folder
3. **Update `curriculum.py`** → Base classes inherit from `StatsLogger`
4. **Major update `learning_progress_algorithm.py`** → Integrate bidirectional scoring directly
5. **Update `__init__.py`** → Remove modules imports, add direct imports
6. **Delete `learning_progress_modules/`** → Clean up folder structure
7. **Update recipes** → Navigation and arena use bidirectional by default

### Benefits of Direct Integration Approach

1. **Cleaner Architecture**
   - No separate modules folder to maintain
   - Core components at appropriate level in curriculum folder
   - Unified stats system available to all algorithms

2. **Better Terminology**
   - SliceAnalyzer more accurately describes parameter space analysis
   - Slice terminology reflects probability distribution analysis

3. **Simpler Imports**
   - Direct imports: `from .task_tracker import TaskTracker`
   - Cleaner `__init__.py` exports

4. **Better Integration**
   - Bidirectional scoring fully integrated into `LearningProgressAlgorithm`
   - `SliceAnalyzer` directly available in `stats.py` for any algorithm
   - `StatsLogger` provides unified interface for all future algorithms

5. **Reduced Complexity**
   - One less folder to navigate
   - Functionality lives where it's most relevant
   - Easier to understand the codebase structure

---

**This plan provides a clean, systematic approach to adapt the diff while maintaining the architectural vision outlined in the branch summary, using better terminology (SliceAnalyzer), and making bidirectional learning progress the default method.**
