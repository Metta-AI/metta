# WandB Metrics Documentation

This directory contains comprehensive documentation for all metrics logged to Weights & Biases (WandB) during
Metta training runs.

## Overview

Our WandB logging captures detailed metrics across multiple categories to monitor training progress, agent
behavior, environment dynamics, and system performance.

## Metric Categories

| Section | Description | Metric Count |
|---------|-------------|--------------|
| [`env_agent/`](./env_agent/) | Detailed agent behavior metrics including actions taken, items collected, combat outcomes,... | 1169 |
| [`env_game/`](./env_game/) | Game object counts and token tracking | 214 |
| [`env_timing_per_epoch/`](./env_timing_per_epoch/) | Performance profiling metrics measured per epoch. Useful for identifying bottlenecks and... | 42 |
| [`eval_memory/`](./eval_memory/) | Metrics for eval memory | 26 |
| [`eval_navigation/`](./eval_navigation/) | Metrics for eval navigation | 26 |
| [`env_task_reward/`](./env_task_reward/) | Task completion rewards | 23 |
| [`env_task_timing/`](./env_task_timing/) | Task initialization timing | 23 |
| [`env_curriculum_task_probs/`](./env_curriculum_task_probs/) | Metrics for env curriculum task probs | 22 |
| [`env_task_completions/`](./env_task_completions/) | Metrics for env task completions | 22 |
| [`env_timing_cumulative/`](./env_timing_cumulative/) | Cumulative environment timing statistics | 22 |
| [`monitor/`](./monitor/) | System resource monitoring | 18 |
| [`trainer_memory/`](./trainer_memory/) | Memory usage by trainer components | 18 |
| [`eval_navsequence/`](./eval_navsequence/) | Metrics for eval navsequence | 17 |
| [`env_attributes/`](./env_attributes/) | Environment configuration and episode attributes | 16 |
| [`eval_objectuse/`](./eval_objectuse/) | Metrics for eval objectuse | 14 |
| [`env_map_reward/`](./env_map_reward/) | Map-specific reward statistics | 11 |
| [`timing_per_epoch/`](./timing_per_epoch/) | Per-epoch training timing | 11 |
| [`experience/`](./experience/) | Training experience buffer statistics | 9 |
| [`losses/`](./losses/) | Training loss components that indicate learning progress and stability. Monitor these to... | 9 |
| [`overview/`](./overview/) | High-level summary metrics that provide a quick assessment of training progress and performance. | 7 |
| [`timing_cumulative/`](./timing_cumulative/) | Cumulative training timing | 7 |
| [`parameters/`](./parameters/) | Training hyperparameters | 5 |
| [`metric/`](./metric/) | Core tracking metrics that serve as x-axis values for other plots. These metrics track... | 4 |
| [`replays/`](./replays/) | Metrics for replays | 1 |
| [`torch_traces/`](./torch_traces/) | Metrics for torch traces | 1 |

**Total Metrics:** 1777

## Metric Aggregation Strategy

Metta uses a multi-stage aggregation pipeline to produce the final metrics logged to WandB:

### Aggregation Pipeline

```
Per-Agent Values → Per-Episode Means → Cross-Episode Means → WandB Logs
```

### Detailed Aggregation Table

| Metric Category | Stage 1: Environment<br>(per episode) | Stage 2: Rollout<br>(collection) | Stage 3: Trainer<br>(final processing) | Final Output |
|----------------|------------------------|------------------|----------------------|--------------|
| **Agent Rewards** | Sum across agents ÷ num_agents | Collect all episode means into list | Mean of all episodes | `env_map_reward/*` = mean<br>`env_map_reward/*.std_dev` = std |
| **Agent Stats**<br>(e.g., actions, items) | Sum across agents ÷ num_agents | Collect all episode values into list | Mean of all episodes | `env_agent/*` = mean<br>`env_agent/*.std_dev` = std |
| **Game Stats**<br>(environment-level) | Single value (no aggregation) | Collect all episode values | Mean of all episodes | `env_game/*` = mean<br>`env_game/*.std_dev` = std |
| **Per-Epoch Timing** | Single value per operation | Keep latest value only | Pass through latest | `env_timing_per_epoch/*` = latest<br>`timing_per_epoch/*` = latest |
| **Cumulative Timing** | Single value per operation | Running average over all steps | Current running average | `env_timing_cumulative/*` = running avg<br>`timing_cumulative/*` = running avg |
| **Attributes**<br>(seed, map size, etc.) | Single value (no aggregation) | Last value overwrites | Pass through | `env_attributes/*` = value |
| **Task Rewards** | Mean across agents | Collect all episode means | Mean of all episodes | `env_task_reward/*` = mean |
| **Curriculum Stats** | Single value | Last value overwrites | Pass through | `env_curriculum/*` = value |

### Timing Metrics Explained

Metta tracks two types of timing metrics:

1. **Per-Epoch Timing** (`*_per_epoch`):
   - Shows the time taken for the most recent epoch/step only
   - Not averaged - each logged value represents that specific step's timing
   - Useful for: Identifying performance changes or spikes in specific steps

2. **Cumulative Timing** (`*_cumulative`):
   - Shows the running average of all steps up to the current point
   - At step N, this is the average of steps 1 through N
   - Useful for: Understanding overall performance trends and smoothing out variance

### Example: Timing Metrics Over 3 Steps

If rollout timing has values [100ms, 150ms, 120ms]:

- **Per-Epoch**:
  - Step 1: `env_timing_per_epoch/rollout` = 100ms
  - Step 2: `env_timing_per_epoch/rollout` = 150ms
  - Step 3: `env_timing_per_epoch/rollout` = 120ms

- **Cumulative**:
  - Step 1: `env_timing_cumulative/rollout` = 100ms (avg of: 100)
  - Step 2: `env_timing_cumulative/rollout` = 125ms (avg of: 100, 150)
  - Step 3: `env_timing_cumulative/rollout` = 123ms (avg of: 100, 150, 120)

### Key Points

1. **Double Averaging**: Most metrics undergo two averaging operations:
   - First: Average across all agents in an episode
   - Second: Average across all episodes in the rollout

2. **Standard Deviation**: The trainer adds `.std_dev` variants showing variance across episodes

3. **Episode Granularity**: Aggregation preserves episode boundaries - partial episodes are not mixed with complete ones

4. **Multi-GPU Training**: Each GPU computes its own statistics independently; WandB handles any cross-GPU aggregation

### Example: Tracing a Reward Metric

Consider `env_map_reward/collectibles` with 4 agents and 3 completed episodes:

1. **Episode 1**: Agents score [2, 3, 1, 4] → Mean: 2.5
2. **Episode 2**: Agents score [3, 3, 2, 2] → Mean: 2.5
3. **Episode 3**: Agents score [1, 2, 3, 2] → Mean: 2.0

**Rollout Collection**: `[2.5, 2.5, 2.0]`

**Final Processing**:
- `env_map_reward/collectibles` = 2.33 (mean)
- `env_map_reward/collectibles.std_dev` = 0.29 (standard deviation)

### Special Cases

- **Diversity Bonus**: Applied to individual agent rewards before any aggregation
- **Kickstarter Losses**: Not aggregated by episode, averaged across all training steps
- **Gradient Stats**: Computed across all parameters, not per-episode

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
./collect_metrics.py <run_id>  # Fetches metrics to wandb_metrics.csv
./generate_docs.py             # Regenerates documentation
```
