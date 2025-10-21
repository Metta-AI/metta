# Learning Progress Config Comparison

## Current Branch vs `msb_currcent_prog`

### Configuration Parameters

| Parameter | Current Branch | `msb_currcent_prog` | Difference |
|-----------|---------------|---------------------|------------|
| **Core Settings** |
| `type` | `"learning_progress"` | `"learning_progress"` | ✅ Same |
| **Bidirectional LP Settings** |
| `use_bidirectional` | `True` | `True` | ✅ Same |
| `ema_timescale` | `0.1` | `0.1` | ✅ Same |
| `slow_timescale_factor` | `0.2` | `0.2` | ✅ Same |
| `exploration_bonus` | `0.1` | `0.1` | ✅ Same |
| `progress_smoothing` | `0.01` | `0.01` | ✅ Same |
| `performance_bonus_weight` | `0.0` | `0.0` | ✅ Same |
| **Task Distribution & Sampling** |
| `num_active_tasks` | `1000` | `10000` | ⚠️ **DIFFERENT: 1000 vs 10000** |
| `rand_task_rate` | `0.01` | `0.01` | ✅ Same |
| `sample_threshold` | `10` | `10` | ✅ Same |
| `memory` | `25` | `25` | ✅ Same |
| `eviction_threshold_percentile` | `0.4` | `0.4` | ✅ Same |
| **Basic EMA Mode Parameters** |
| `basic_ema_initial_alpha` | `0.3` | `0.3` | ✅ Same |
| `basic_ema_alpha_decay` | `0.2` | `0.2` | ✅ Same |
| `exploration_blend_factor` | `0.5` | `0.5` | ✅ Same |
| **Task Tracker EMA** |
| `task_tracker_ema_alpha` | `0.02` | `0.02` | ✅ Same |
| **Task Creation Defaults** |
| `task_default_success_threshold` | `0.5` | `0.5` | ✅ Same |
| `task_default_generator_type` | `0.0` | `0.0` | ✅ Same |
| **Performance & Memory Management** |
| `max_slice_axes` | `3` | `3` | ✅ Same |
| **Memory Backend Configuration** |
| `task_struct_size` | `13` | `13` | ✅ Same |
| `completion_history_size` | `1000` | `1000` | ✅ Same |
| `enable_detailed_slice_logging` | `False` | `False` | ✅ Same |
| `use_shared_memory` | `True` | `True` | ✅ Same |
| `session_id` | `None` | `None` | ✅ Same |

## Summary

### Parameters Present in Both Branches: ✅ All parameters present

All configuration parameters are present in both branches.

### Default Value Differences: 1

**Only difference:**
- **`num_active_tasks`**: Current branch has `1000`, `msb_currcent_prog` has `10000`
  - This is a 10x difference in the number of active tasks maintained in the curriculum pool
  - The current branch value of `1000` was intentionally reduced (as noted in the `task_generator.py` where it changed from 16 to 1000)
  - This likely reflects a trade-off between:
    - **Memory efficiency** (smaller pool = less memory)
    - **Task diversity** (larger pool = more diverse tasks)
    - **Computational efficiency** (smaller pool = faster scoring/updates)

### Impact Analysis

The `num_active_tasks` parameter affects:
1. **Memory usage** in both local and shared memory backends
2. **Task pool diversity** - how many different tasks are actively tracked
3. **Learning progress calculation** - more tasks = more statistical stability but more overhead
4. **Eviction frequency** - smaller pool = more frequent task eviction
5. **Default value in `CurriculumConfig`** - syncs to match this value

### Recommendation

The current branch's value of `1000` appears to be an intentional optimization for:
- Better performance in most training scenarios
- More reasonable memory footprint
- Still maintains good task diversity for curriculum learning

If you need the larger task pool (10000), you can override it via:
```python
config = LearningProgressConfig(num_active_tasks=10000)
```

