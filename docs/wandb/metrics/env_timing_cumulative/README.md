# Env Timing Cumulative Metrics

## Overview

Cumulative environment timing statistics

**Total metrics in this section:** 22

## Metric Suffixes

This section contains metrics with the following statistical suffixes:

- **`.std_dev`** - Standard deviation across episodes (variance measure)
  - Formula: `sqrt(sum((x - mean)²) / n)`

## Subsections

### Active Frac

**Count:** 20 metrics

**_c_env.get_episode_stats:** (1 value / 1 std_dev)
- `env_timing_cumulative/active_frac/_c_env.get_episode_stats`
- `env_timing_cumulative/active_frac/_c_env.get_episode_stats.std_dev`

**_c_env.step:** (1 value / 1 std_dev)
- `env_timing_cumulative/active_frac/_c_env.step`
- `env_timing_cumulative/active_frac/_c_env.step.std_dev`

**_initialize_c_env:** (1 value / 1 std_dev)
- `env_timing_cumulative/active_frac/_initialize_c_env`
- `env_timing_cumulative/active_frac/_initialize_c_env.std_dev`

**_initialize_c_env.build_map:** (1 value / 1 std_dev)
- `env_timing_cumulative/active_frac/_initialize_c_env.build_map`
- `env_timing_cumulative/active_frac/_initialize_c_env.build_map.std_dev`

**_initialize_c_env.make_c_env:** (1 value / 1 std_dev)
- `env_timing_cumulative/active_frac/_initialize_c_env.make_c_env`
- `env_timing_cumulative/active_frac/_initialize_c_env.make_c_env.std_dev`

**_replay_writer:** (1 value / 1 std_dev)
- `env_timing_cumulative/active_frac/_replay_writer`
- `env_timing_cumulative/active_frac/_replay_writer.std_dev`

**_stats_writer:** (1 value / 1 std_dev)
- `env_timing_cumulative/active_frac/_stats_writer`
- `env_timing_cumulative/active_frac/_stats_writer.std_dev`

**process_episode_stats:** (1 value / 1 std_dev)
- `env_timing_cumulative/active_frac/process_episode_stats`
- `env_timing_cumulative/active_frac/process_episode_stats.std_dev`

**reset:** (1 value / 1 std_dev)
- `env_timing_cumulative/active_frac/reset`
- `env_timing_cumulative/active_frac/reset.std_dev`

**step:** (1 value / 1 std_dev)
- `env_timing_cumulative/active_frac/step`
- `env_timing_cumulative/active_frac/step.std_dev`


### Frac

**Count:** 2 metrics

**thread_idle:** (1 value / 1 std_dev)
- `env_timing_cumulative/frac/thread_idle`
- `env_timing_cumulative/frac/thread_idle.std_dev`


