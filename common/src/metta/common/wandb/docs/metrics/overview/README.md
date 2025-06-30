# Overview Metrics

## Overview

High-level training progress

**Total metrics in this section:** 3

## Subsections

### General Metrics

**Count:** 3 metrics

**reward:**
- `overview/reward`

The mean reward achieved per episode per agent, averaged over all environments and agents over the previous epoch.

**Interpretation:** Higher values indicate better agent performance. Monitor for consistent improvement during training.

**reward_vs_total_time:**
- `overview/reward_vs_total_time`

Reward plotted against total training time, showing learning efficiency. (Unit: reward per second)

**Interpretation:** Steeper curves indicate faster learning. Plateaus may suggest convergence or need for hyperparameter adjustment.

**sps:**
- `overview/sps`

Steps per second - the throughput of the training system. (Unit: steps/second)

**Interpretation:** Higher is better. Drops may indicate resource contention or environment complexity changes.


