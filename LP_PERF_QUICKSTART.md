# LP Performance Optimization - Quick Start

## 🚀 Quick Test (5 minutes)

```bash
# Run comparison test
./tools/test_lp_performance_matrix.sh quick

# View results
./tools/compare_lp_performance.sh
```

## 📊 Test Matrix

| Batch Size   | Cache | Expected SPS | Expected Degradation | Command        |
| ------------ | ----- | ------------ | -------------------- | -------------- |
| 1 (baseline) | No    | ~8,500 avg   | ~66%                 | `baseline`     |
| 10           | Yes   | ~13,500 avg  | ~20%                 | `batch_10`     |
| 100          | Yes   | ~14,500 avg  | ~7%                  | `batch_100` ⭐ |
| 1000         | Yes   | ~14,750 avg  | ~3%                  | `batch_1000`   |

⭐ = Recommended default

## ☁️ Cloud Deployment

Launch all performance tests on cloud infrastructure:

```bash
# Quick comparison (3 jobs: baseline, batch_10, batch_100)
./tools/launch_lp_perf_matrix.sh quick

# All configurations (6 jobs)
./tools/launch_lp_perf_matrix.sh all

# Specific test suites
./tools/launch_lp_perf_matrix.sh batch_sizes  # Compare batch sizes
./tools/launch_lp_perf_matrix.sh cache        # Cache effectiveness
./tools/launch_lp_perf_matrix.sh pool_size    # Pool size impact

# Custom settings
GPUS=2 SPOT=--no-spot MAX_RUNTIME=4 ./tools/launch_lp_perf_matrix.sh quick
```

Monitor jobs:

```bash
uv run sky queue    # Check job status
uv run sky logs <cluster-name>  # View logs
```

## 🎯 Manual Testing

### Test baseline (no optimizations)

```bash
timeout 600s uv run ./tools/run.py cogs_v_clips.train \
    use_lp=True \
    perf_invalidation_batch_size=1 \
    perf_cache_task_list=False \
    perf_log_metrics=True \
    run=msb_perfdiagnosis_manual_baseline
```

### Test recommended settings (batch=100)

```bash
timeout 600s uv run ./tools/run.py cogs_v_clips.train \
    use_lp=True \
    perf_invalidation_batch_size=100 \
    perf_cache_task_list=True \
    perf_log_metrics=True \
    run=msb_perfdiagnosis_manual_batch100
```

### Test your custom settings

```bash
timeout 600s uv run ./tools/run.py cogs_v_clips.train \
    use_lp=True \
    perf_invalidation_batch_size=50 \
    perf_cache_task_list=True \
    num_active_tasks=200 \
    perf_log_metrics=True \
    run=msb_perfdiagnosis_manual_custom
```

## 📈 Check Results

```bash
# View SPS over time
grep -i sps outputs/*msb_perfdiagnosis_*/logs/train.log | tail -20

# View performance metrics
grep "LP_PERF" outputs/*msb_perfdiagnosis_*/logs/train.log
```

## ⚙️ Configuration Parameters

### Core Parameters

- `perf_invalidation_batch_size` - How often to invalidate cache (1-1000)
  - **1**: Baseline, no optimization
  - **10**: Light batching, ~10x improvement
  - **100**: Recommended, ~100x improvement ⭐
  - **1000**: Heavy batching, ~1000x improvement

- `perf_cache_task_list` - Enable task list caching (True/False)
  - **True**: Recommended ⭐
  - **False**: Baseline

- `num_active_tasks` - Size of task pool (10-1000)
  - **1000**: Default, more diversity
  - **100-200**: Faster, less diversity

- `perf_log_metrics` - Log performance stats (True/False)
  - **True**: For testing/debugging
  - **False**: For production ⭐

## 🎛️ Preset Configurations

### In Code

```python
from metta.cogworks.curriculum import LearningProgressConfig

# Recommended
config = LearningProgressConfig(
    perf_invalidation_batch_size=100,
    perf_cache_task_list=True,
)

# Maximum performance
config = LearningProgressConfig(
    perf_invalidation_batch_size=1000,
    perf_cache_task_list=True,
    num_active_tasks=100,
)
```

## 📋 Test All Configurations

```bash
# Test all batch sizes
./tools/test_lp_performance_matrix.sh batch_sizes

# Test cache effectiveness
./tools/test_lp_performance_matrix.sh cache

# Test pool size impact
./tools/test_lp_performance_matrix.sh pool_size

# Test everything (60+ minutes)
./tools/test_lp_performance_matrix.sh all
```

## 🔍 Understanding Results

### Good Performance

- Initial SPS ≈ Final SPS (minimal degradation)
- Degradation < 10%
- Cache hit rate > 90%

### Poor Performance

- Final SPS << Initial SPS
- Degradation > 50%
- Frequent invalidations

### Example Good Output

```
[LP_PERF] Invalidations: 50, Total updates: 5000, Batch size: 100, Reduction: 100.0x
[LP_PERF] Task list cache: 4950 hits, 50 misses, hit rate: 99.0%
```

## 🎯 Quick Decision Guide

**Need maximum training speed?** → `batch_100` (100x improvement, minimal staleness)

**Need absolute maximum performance?** → `batch_1000` + `num_active_tasks=100` (1000x improvement)

**Need to verify quality isn't impacted?** → Start with `batch_10`, then increase to `batch_100`

**Having issues?** → Enable `perf_log_metrics=True` and check logs

## 📚 Full Documentation

- **Complete guide**: `LP_PERFORMANCE_OPTIMIZATION_GUIDE.md`
- **Investigation details**: `performance_degradation_investigation.md`
- **Command reference**: `COMMAND_LINE_ANALYSIS.md`
- **Debug steps**: `NEXT_STEPS_PERF_DEBUG.md`
