# WandB Metrics Documentation

This directory contains comprehensive documentation for all metrics logged to Weights & Biases (WandB) during
Metta training runs.

## Overview

Our WandB logging captures detailed metrics across multiple categories to monitor training progress, agent
behavior, environment dynamics, and system performance.

## Metric Categories

| Section | Description | Metric Count |
|---------|-------------|--------------|
| [`env_agent/`](./env_agent/) | Agent actions, rewards, and item interactions | 1120 |
| [`env_game/`](./env_game/) | Game object counts and token tracking | 214 |
| [`env_timing_per_epoch/`](./env_timing_per_epoch/) | Per-epoch environment timing breakdown | 42 |
| [`env_timing_cumulative/`](./env_timing_cumulative/) | Cumulative environment timing statistics | 22 |
| [`monitor/`](./monitor/) | System resource monitoring | 18 |
| [`trainer_memory/`](./trainer_memory/) | Memory usage by trainer components | 18 |
| [`env_attributes/`](./env_attributes/) | Environment configuration and episode attributes | 16 |
| [`experience/`](./experience/) | Training experience buffer statistics | 9 |
| [`timing_per_epoch/`](./timing_per_epoch/) | Per-epoch training timing | 9 |
| [`losses/`](./losses/) | Training loss components | 8 |
| [`parameters/`](./parameters/) | Training hyperparameters | 5 |
| [`timing_cumulative/`](./timing_cumulative/) | Cumulative training timing | 5 |
| [`env_map_reward/`](./env_map_reward/) | Map-specific reward statistics | 4 |
| [`metric/`](./metric/) | Core training metrics (steps, epochs, time) | 4 |
| [`overview/`](./overview/) | High-level training progress | 3 |
| [`env_task_reward/`](./env_task_reward/) | Task completion rewards | 2 |
| [`env_task_timing/`](./env_task_timing/) | Task initialization timing | 2 |

**Total Metrics:** 1501

## Metric Naming Convention

Metrics follow a hierarchical naming structure:
```
section/subsection/metric_name[.statistic][.qualifier]
```

### Common Statistics Suffixes
- `.avg` - Average value
- `.std_dev` - Standard deviation
- `.min` - Minimum value
- `.max` - Maximum value
- `.first_step` - First step where metric was recorded
- `.last_step` - Last step where metric was recorded
- `.rate` - Rate of occurrence
- `.updates` - Number of updates
- `.activity_rate` - Fraction of time the metric was active

### Common Qualifiers
- `.agent` - Per-agent breakdown
- `.success` / `.failed` - Outcome-specific metrics
- `.gained` / `.lost` - Change tracking

## Usage

Each subdirectory contains:
- `README.md` - Detailed documentation for that metric category
- Explanations of what each metric measures
- Relationships between related metrics
- Tips for interpretation and debugging

## Quick Start

To explore specific metric categories:
1. Navigate to the relevant subdirectory
2. Read the README for detailed explanations
3. Use the metric names when querying WandB or analyzing logs

## Related Tools

- [`collect_metrics.py`](../../collect_metrics.py) - Script to fetch metrics from WandB runs
- [`generate_docs.py`](../../generate_docs.py) - Script to generate this documentation

## Updating Documentation

To update this documentation with metrics from a new run:
```bash
cd common/src/metta/common/wandb
./collect_metrics.py <run_id>  # Fetches metrics to wandb_metrics.txt
./generate_docs.py             # Regenerates documentation
```
