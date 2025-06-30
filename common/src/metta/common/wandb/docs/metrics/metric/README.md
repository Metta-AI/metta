# Metric Metrics

## Overview

Core training metrics (steps, epochs, time)

**Total metrics in this section:** 4

## Subsections

### General Metrics

**Count:** 4 metrics

**agent_step:**
- `metric/agent_step`

Total number of agent steps taken across all environments.

**Interpretation:** Primary measure of training progress. Compare with wall-clock time for efficiency.

**epoch:**
- `metric/epoch`

Current training epoch number.

**Interpretation:** One epoch represents a full cycle of experience collection and training.

**total_time:**
- `metric/total_time`

Total wall-clock time since training started. (Unit: seconds)

**train_time:**
- `metric/train_time`

Time spent in the training/optimization phase, excluding environment rollouts. (Unit: seconds)

**Interpretation:** Compare with total_time to understand training vs rollout balance.


