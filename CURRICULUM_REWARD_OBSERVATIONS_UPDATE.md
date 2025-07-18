# Curriculum Reward Observations Update Summary

This document summarizes the changes made to update the curriculum/bbc to use reward observations.

## Overview

The BBC (Behavior-Based Curriculum) using Learning Progress has been updated to support detailed reward observations instead of relying solely on aggregated episode rewards. This allows the curriculum to track learning progress for each reward type independently, enabling more sophisticated task selection strategies.

## Key Changes

### 1. Core Curriculum Interface Updates

**File: `mettagrid/src/metta/mettagrid/curriculum/core.py`**
- Updated `Curriculum.complete_task()` and `Task.complete()` to accept `Union[float, Dict[str, float]]`
- Maintains backward compatibility with single float scores

### 2. Learning Progress Curriculum Enhancement

**File: `mettagrid/src/metta/mettagrid/curriculum/learning_progress.py`**
- Added support for tracking multiple reward types independently
- New parameters:
  - `use_reward_observations`: Enable/disable reward observation tracking
  - `reward_types`: List of reward types to track
  - `reward_aggregation`: Method to combine progress ("mean", "max", "sum")
- Each reward type gets its own `BidirectionalLearningProgress` tracker
- Statistics are now reported per reward type

### 3. Environment Updates

**Files: `mettagrid/src/metta/mettagrid/base_env.py` and `mettagrid/src/metta/mettagrid/mettagrid_env.py`**
- Modified episode completion to extract detailed reward observations from agent stats
- Looks for ".gained" statistics (e.g., "ore_red.gained", "battery_red.gained")
- Normalizes values by number of agents
- Passes dictionary of reward observations to curriculum when enabled

### 4. Other Curriculum Updates

Updated all curriculum classes that override `complete_task` to handle the new format:
- **`multi_task.py`**: Updated type signature
- **`random.py`**: Added Union import
- **`progressive.py`**: Both `ProgressiveCurriculum` and `ProgressiveMultiTaskCurriculum`
- **`prioritize_regressed.py`**: Handles dict scores by using "total" or averaging

### 5. Configuration Updates

**File: `configs/env/mettagrid/curriculum/bbc/bbc.yaml`**
- Added reward observation parameters with sensible defaults
- Tracks all major reward types: ore_red, battery_red, heart, laser, armor, blueprint

### 6. Example Configurations

Created example configurations demonstrating different use cases:
- **`bbc_reward_max.yaml`**: Uses max aggregation to prioritize any improvement
- **`bbc_combat_focus.yaml`**: Tracks only combat-related rewards

### 7. Documentation

**File: `configs/env/mettagrid/curriculum/bbc/REWARD_OBSERVATIONS.md`**
- Comprehensive documentation of the feature
- Configuration examples
- Use cases and benefits

## Benefits

1. **Granular Progress Tracking**: Track learning progress for each objective separately
2. **Flexible Task Selection**: Different aggregation methods for different training goals
3. **Better Insights**: Detailed statistics show which rewards agents are improving on
4. **Backward Compatible**: Existing code continues to work without changes

## Usage

To use reward observations, update your curriculum configuration:

```yaml
_target_: metta.mettagrid.curriculum.learning_progress.LearningProgressCurriculum
use_reward_observations: true
reward_types: ["ore_red", "battery_red", "heart", "laser", "armor", "blueprint"]
reward_aggregation: "mean"
```

The curriculum will automatically track learning progress for each reward type and use the aggregated progress to select tasks.

## Future Enhancements

Possible future improvements:
- Per-reward-type weights for weighted aggregation
- Dynamic reward type discovery from environment
- Reward-specific task selection thresholds
- Integration with other curriculum types (bucketed, etc.)