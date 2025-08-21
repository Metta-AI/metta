# Metric Metrics

## Overview

Core tracking metrics that serve as x-axis values for other plots. These metrics track
fundamental progress indicators like steps, epochs, and time. When analyzing other metrics,
these values provide the temporal context for understanding changes over the training run.

**Total metrics in this section:** 4

## Subsections

### General Metrics

**Count:** 4 metrics

**agent_step:** (1 value)
- `metric/agent_step`
  - Total number of agent steps taken across all environments. This is the primary x-axis value
    for most training curves as it represents actual experience collected.
  - **Interpretation:** Use this as x-axis when comparing runs with different environment counts or speeds.


**epoch:** (1 value)
- `metric/epoch`
  - Current training epoch number. Each epoch represents one cycle of rollout collection and training.
    Useful as x-axis for metrics that update once per epoch.
  - **Interpretation:** Good for comparing architectural changes where steps-per-epoch may vary.


**total_time:** (1 value)
- `metric/total_time`
  - Total wall-clock time since training started. Use as x-axis when comparing real-world efficiency
    or when coordinating with external events. (Unit: seconds)


**train_time:** (1 value)
- `metric/train_time`
  - Time spent in the training/optimization phase, excluding environment rollouts. (Unit: seconds)
  - **Interpretation:** Compare with total_time to understand training vs rollout balance.



