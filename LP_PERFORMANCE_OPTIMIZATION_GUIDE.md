# LP Performance Optimization Guide

## Overview

This guide describes the comprehensive performance optimization system for Learning Progress curriculum that allows
testing multiple optimization strategies with configurable parameters.

## Problem Summary

Learning Progress curriculum experiences performance degradation over time due to:

1. **`get_all_tracked_tasks()` called on every episode** → O(max_tasks) memory scans
2. **No caching** of task lists → repeated expensive operations
3. With 1000 max_tasks × 5000 episodes/epoch = **5M memory operations per epoch**

## Solutions Implemented

### 1. **Batch Cache Invalidation** (Primary Fix)

**Parameter**: `perf_invalidation_batch_size`

Instead of invalidating LP score cache on every episode, batch the invalidations:

- `1` = Baseline (original behavior, invalidate every update)
- `10` = Light batching (invalidate every 10 updates)
- `100` = Recommended (invalidate every 100 updates)
- `1000` = Heavy batching (invalidate every 1000 updates)

**Impact**: Reduces `get_all_tracked_tasks()` calls from ~5000/epoch to ~50/epoch with batch_size=100 (100x reduction)

### 2. **Task List Caching**

**Parameter**: `perf_cache_task_list` (bool)

Cache the result of `get_all_tracked_tasks()` and only invalidate when tasks are created/removed:

- Cache is automatically invalidated on task creation/eviction
- Tracks cache hit rate for monitoring
- Most effective when combined with batch invalidation

**Impact**: Converts repeated O(max_tasks) scans to O(1) cache hits

### 3. **Reduced Pool Size**

**Parameter**: `num_active_tasks`

Use fewer tasks for faster operations:

- Default: 1000 tasks
- Small: 100-200 tasks
- Trade-off: Faster sampling vs less curriculum diversity

### 4. **Performance Monitoring**

**Parameter**: `perf_log_metrics` (bool)

Enables detailed logging:

- Invalidation counts and reduction ratios
- Cache hit rates
- Helps understand which optimizations are effective

## Test Configurations

### Quick Comparison

```bash
./tools/test_lp_performance_matrix.sh quick
```

Tests: baseline, batch_10, batch_100

### All Configurations

```bash
./tools/test_lp_performance_matrix.sh all
```

Available test configurations: | Configuration | Batch Size | Cache | Pool Size | Description |
|--------------|------------|-------|-----------|-------------| | `baseline` | 1 | No | 1000 | Original behavior (for
comparison) | | `batch_10` | 10 | Yes | 1000 | Light batching | | `batch_100` | 100 | Yes | 1000 | Recommended setting |
| `batch_1000` | 1000 | Yes | 1000 | Heavy batching | | `cache_only` | 1 | Yes | 1000 | Cache without batching | |
`small_pool_100` | 100 | Yes | 100 | Small pool + batching |

### Test Specific Strategies

```bash
# Test different batch sizes
./tools/test_lp_performance_matrix.sh batch_sizes

# Test cache effectiveness
./tools/test_lp_performance_matrix.sh cache

# Test pool size impact
./tools/test_lp_performance_matrix.sh pool_size

# Run a specific test
./tools/test_lp_performance_matrix.sh batch_100
```

## Manual Configuration

### Option 1: Using Config Parameters

```bash
# Baseline (no optimizations)
uv run ./tools/run.py cogs_v_clips.train \
    use_lp=True \
    perf_invalidation_batch_size=1 \
    perf_cache_task_list=False \
    run=test_baseline

# Recommended settings
uv run ./tools/run.py cogs_v_clips.train \
    use_lp=True \
    perf_invalidation_batch_size=100 \
    perf_cache_task_list=True \
    perf_log_metrics=True \
    run=test_optimized

# Custom combination
uv run ./tools/run.py cogs_v_clips.train \
    use_lp=True \
    perf_invalidation_batch_size=50 \
    perf_cache_task_list=True \
    num_active_tasks=200 \
    perf_log_metrics=True \
    run=test_custom
```

### Option 2: In Recipe Code

```python
from metta.cogworks.curriculum import LearningProgressConfig

# Baseline
config = LearningProgressConfig(
    perf_invalidation_batch_size=1,
    perf_cache_task_list=False,
)

# Recommended
config = LearningProgressConfig(
    perf_invalidation_batch_size=100,
    perf_cache_task_list=True,
    perf_log_metrics=True,
)

# Aggressive optimization
config = LearningProgressConfig(
    perf_invalidation_batch_size=1000,
    perf_cache_task_list=True,
    num_active_tasks=100,
    perf_log_metrics=True,
)
```

## Comparing Results

After running tests, compare performance:

```bash
./tools/compare_lp_performance.sh
```

This generates a table showing:

- Initial SPS, Final SPS, Average SPS
- Degradation percentage
- Cache hit rates
- Invalidation reduction ratios
- Recommendation for best configuration

Example output:

```
Configuration        Initial SPS   Final SPS     Avg SPS  Degradation %   Samples
--------------------------------------------------------------------------------
baseline                 15000.0      5000.0     8500.0           66.7%        20
batch_10                 15000.0     12000.0    13500.0           20.0%        20
batch_100                15000.0     14000.0    14500.0            6.7%        20
batch_1000               15000.0     14500.0    14750.0            3.3%        20
cache_only               15000.0      6000.0     9000.0           60.0%        20
small_pool_100           16000.0     15500.0    15750.0            3.1%        20
```

## Expected Results

### Baseline (No Optimizations)

- **Initial SPS**: ~15,000
- **Final SPS**: ~5,000
- **Degradation**: ~66%
- **Cause**: 5,000 memory scans per epoch

### Batch Size = 100 (Recommended)

- **Initial SPS**: ~15,000
- **Final SPS**: ~14,000
- **Degradation**: ~7%
- **Improvement**: 100x fewer memory scans

### Small Pool (100 tasks)

- **Initial SPS**: ~16,000
- **Final SPS**: ~15,500
- **Degradation**: ~3%
- **Trade-off**: Less curriculum diversity

## Performance Monitoring

When `perf_log_metrics=True`, you'll see logs like:

```
[LP_PERF] Invalidations: 50, Total updates: 5000, Batch size: 100, Reduction: 100.0x
[LP_PERF] Task list cache: 4950 hits, 50 misses, hit rate: 99.0%
```

This helps verify optimizations are working as expected.

## Trade-offs

### Batch Invalidation

**Pro**: Massive performance improvement **Con**: LP scores can be up to N episodes stale **Recommendation**: Start with
100, tune based on training quality

### Task List Caching

**Pro**: Near-free performance win **Con**: None (cache automatically invalidated when needed) **Recommendation**:
Always enable

### Reduced Pool Size

**Pro**: Faster operations, less memory **Con**: Less curriculum diversity **Recommendation**: Use only if 100-200 tasks
sufficient for your problem

## Recommended Settings

For most use cases:

```python
LearningProgressConfig(
    perf_invalidation_batch_size=100,  # 100x reduction in scans
    perf_cache_task_list=True,         # Enable caching
    perf_log_metrics=False,            # Disable in production
)
```

For maximum performance:

```python
LearningProgressConfig(
    perf_invalidation_batch_size=1000,  # 1000x reduction
    perf_cache_task_list=True,
    num_active_tasks=100,               # Small pool
    perf_log_metrics=False,
)
```

For debugging/testing:

```python
LearningProgressConfig(
    perf_invalidation_batch_size=10,   # Frequent updates for testing
    perf_cache_task_list=True,
    perf_log_metrics=True,              # Detailed logging
)
```

## Files Modified

### Core Implementation

- `metta/cogworks/curriculum/learning_progress_algorithm.py`
  - Added configurable batch invalidation
  - Added performance monitoring
- `metta/cogworks/curriculum/task_tracker.py`
  - Added task list caching
  - Cache invalidation on create/remove
  - Cache hit rate tracking

### Configuration

- `metta/cogworks/curriculum/performance_config.py`
  - Preset configurations
  - Parameter documentation

### Testing Tools

- `tools/test_lp_performance_matrix.sh`
  - Automated testing of all configurations
  - Quick and comprehensive test modes
- `tools/compare_lp_performance.sh`
  - Result comparison and analysis
  - Automatic recommendations

## Next Steps

1. **Run quick test** to confirm improvements:

   ```bash
   ./tools/test_lp_performance_matrix.sh quick
   ```

2. **Compare results**:

   ```bash
   ./tools/compare_lp_performance.sh
   ```

3. **Choose optimal settings** based on your requirements:
   - Training quality vs speed trade-off
   - Pool size requirements
   - Acceptable staleness level

4. **Update production configs** with chosen settings

5. **Monitor** performance metrics to ensure optimizations are effective

## Troubleshooting

### Performance still degrading

- Check `perf_log_metrics` output to verify optimizations are active
- Try smaller `num_active_tasks` (100-200)
- Increase `perf_invalidation_batch_size` (500-1000)

### Training quality degraded

- Reduce `perf_invalidation_batch_size` (50-100)
- Ensure `perf_cache_task_list=True` (no quality impact)
- Check that task pool size is sufficient for your curriculum

### Cache not working

- Verify `perf_cache_task_list=True` in config
- Check logs for cache hit rate (should be >90%)
- Ensure using shared memory backend (`use_shared_memory=True`)

## Support

For questions or issues:

- See `performance_degradation_investigation.md` for full analysis
- See `COMMAND_LINE_ANALYSIS.md` for command examples
- See `NEXT_STEPS_PERF_DEBUG.md` for detailed debugging steps
