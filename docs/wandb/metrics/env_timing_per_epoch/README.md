# Env Timing Per Epoch Metrics

## Overview

Performance profiling metrics measured per epoch. Useful for identifying bottlenecks and
optimizing environment execution speed.

**Total metrics in this section:** 42

## Metric Suffixes

This section contains metrics with the following statistical suffixes:

- **`.std_dev`** - Standard deviation across episodes (variance measure)
  - Formula: `sqrt(sum((x - mean)²) / n)`

## Subsections

### Active Frac

**Count:** 20 metrics

**_c_env.get_episode_stats:** (1 value / 1 std_dev)
- `env_timing_per_epoch/active_frac/_c_env.get_episode_stats`
- `env_timing_per_epoch/active_frac/_c_env.get_episode_stats.std_dev`

**_c_env.step:** (1 value / 1 std_dev)
- `env_timing_per_epoch/active_frac/_c_env.step`
- `env_timing_per_epoch/active_frac/_c_env.step.std_dev`

**_initialize_c_env:** (1 value / 1 std_dev)
- `env_timing_per_epoch/active_frac/_initialize_c_env`
- `env_timing_per_epoch/active_frac/_initialize_c_env.std_dev`

**_initialize_c_env.build_map:** (1 value / 1 std_dev)
- `env_timing_per_epoch/active_frac/_initialize_c_env.build_map`
- `env_timing_per_epoch/active_frac/_initialize_c_env.build_map.std_dev`

**_initialize_c_env.make_c_env:** (1 value / 1 std_dev)
- `env_timing_per_epoch/active_frac/_initialize_c_env.make_c_env`
- `env_timing_per_epoch/active_frac/_initialize_c_env.make_c_env.std_dev`

**_replay_writer:** (1 value / 1 std_dev)
- `env_timing_per_epoch/active_frac/_replay_writer`
- `env_timing_per_epoch/active_frac/_replay_writer.std_dev`

**_stats_writer:** (1 value / 1 std_dev)
- `env_timing_per_epoch/active_frac/_stats_writer`
- `env_timing_per_epoch/active_frac/_stats_writer.std_dev`

**process_episode_stats:** (1 value / 1 std_dev)
- `env_timing_per_epoch/active_frac/process_episode_stats`
- `env_timing_per_epoch/active_frac/process_episode_stats.std_dev`

**reset:** (1 value / 1 std_dev)
- `env_timing_per_epoch/active_frac/reset`
- `env_timing_per_epoch/active_frac/reset.std_dev`

**step:** (1 value / 1 std_dev)
- `env_timing_per_epoch/active_frac/step`
- `env_timing_per_epoch/active_frac/step.std_dev`


### Frac

**Count:** 2 metrics

**thread_idle:** (1 value / 1 std_dev)
- `env_timing_per_epoch/frac/thread_idle`
  - Fraction of time worker threads spend idle.
  - **Interpretation:** High values (>0.9) suggest CPU underutilization. Consider more environments.

- `env_timing_per_epoch/frac/thread_idle.std_dev`
  - Fraction of time worker threads spend idle. (standard deviation)



### Msec

**Count:** 20 metrics

**_c_env.get_episode_stats:** (1 value / 1 std_dev)
- `env_timing_per_epoch/msec/_c_env.get_episode_stats`
- `env_timing_per_epoch/msec/_c_env.get_episode_stats.std_dev`

**_c_env.step:** (1 value / 1 std_dev)
- `env_timing_per_epoch/msec/_c_env.step`
- `env_timing_per_epoch/msec/_c_env.step.std_dev`

**_initialize_c_env:** (1 value / 1 std_dev)
- `env_timing_per_epoch/msec/_initialize_c_env`
- `env_timing_per_epoch/msec/_initialize_c_env.std_dev`

**_initialize_c_env.build_map:** (1 value / 1 std_dev)
- `env_timing_per_epoch/msec/_initialize_c_env.build_map`
- `env_timing_per_epoch/msec/_initialize_c_env.build_map.std_dev`

**_initialize_c_env.make_c_env:** (1 value / 1 std_dev)
- `env_timing_per_epoch/msec/_initialize_c_env.make_c_env`
- `env_timing_per_epoch/msec/_initialize_c_env.make_c_env.std_dev`

**_replay_writer:** (1 value / 1 std_dev)
- `env_timing_per_epoch/msec/_replay_writer`
- `env_timing_per_epoch/msec/_replay_writer.std_dev`

**_stats_writer:** (1 value / 1 std_dev)
- `env_timing_per_epoch/msec/_stats_writer`
- `env_timing_per_epoch/msec/_stats_writer.std_dev`

**process_episode_stats:** (1 value / 1 std_dev)
- `env_timing_per_epoch/msec/process_episode_stats`
- `env_timing_per_epoch/msec/process_episode_stats.std_dev`

**reset:** (1 value / 1 std_dev)
- `env_timing_per_epoch/msec/reset`
- `env_timing_per_epoch/msec/reset.std_dev`

**step:** (1 value / 1 std_dev)
- `env_timing_per_epoch/msec/step`
  - Average milliseconds per environment step in the epoch. (Unit: milliseconds)
  - **Interpretation:** Lower is better. Spikes indicate environment bottlenecks.

- `env_timing_per_epoch/msec/step.std_dev`
  - Average milliseconds per environment step in the epoch. (Unit: milliseconds) (standard deviation)



