# Env Timing Per Epoch Metrics

## Overview

Per-epoch environment timing breakdown

**Total metrics in this section:** 42

## Subsections

### Active Frac

**Count:** 20 metrics

**Metric Groups:**
- `_c_env.get_episode_stats` (2 metrics)
- `_c_env.step` (2 metrics)
- `_initialize_c_env` (2 metrics)
- `_initialize_c_env.build_map` (2 metrics)
- `_initialize_c_env.make_c_env` (2 metrics)
- `_replay_writer` (2 metrics)
- `_stats_writer` (2 metrics)
- `process_episode_stats` (2 metrics)
- `reset` (2 metrics)
- `step` (2 metrics)

### Frac

**Count:** 2 metrics

**thread_idle:**
- `env_timing_per_epoch/frac/thread_idle`
- `env_timing_per_epoch/frac/thread_idle.std_dev`


### Msec

**Count:** 20 metrics

**Metric Groups:**
- `_c_env.get_episode_stats` (2 metrics)
- `_c_env.step` (2 metrics)
- `_initialize_c_env` (2 metrics)
- `_initialize_c_env.build_map` (2 metrics)
- `_initialize_c_env.make_c_env` (2 metrics)
- `_replay_writer` (2 metrics)
- `_stats_writer` (2 metrics)
- `process_episode_stats` (2 metrics)
- `reset` (2 metrics)
- `step` (2 metrics)


## Interpretation Guide

### Timing Categories
- `msec/` - Raw millisecond timings
- `frac/` - Fraction of total time
- `active_frac/` - Fraction of active (non-idle) time

### Key Operations
- `step` - Environment step execution
- `reset` - Episode reset operations
- `_initialize_c_env` - C++ environment initialization
- `process_episode_stats` - Statistics processing

### Performance Analysis
1. **Bottlenecks**: Look for operations with high `msec` values
2. **Efficiency**: Check `thread_idle` for CPU utilization
3. **Variability**: High `std_dev` values indicate inconsistent performance
