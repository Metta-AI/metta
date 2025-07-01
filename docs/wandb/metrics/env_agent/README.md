# Env Agent Metrics

## Overview

Detailed agent behavior metrics including actions taken, items collected, combat outcomes,
and other agent-specific events. These metrics help understand what agents are actually doing
in the environment.

**Total metrics in this section:** 1164

## Metric Suffixes

This section contains metrics with the following statistical suffixes:

- **`.activity_rate`** - Frequency during active period only (updates per step between first and last occurrence)
  - Formula: `(update_count - 1) / (last_step - first_step)`
  - Note: Subtracts 1 because the first update just marks the start of activity
- **`.avg`** - Average value of the metric across updates within an episode
  - Formula: `sum(values) / update_count`
- **`.first_step`** - First step where this metric was recorded
- **`.last_step`** - Last step where this metric was recorded
- **`.max`** - Maximum value observed during the episode
- **`.min`** - Minimum value observed during the episode
- **`.rate`** - Frequency of updates (updates per step over entire episode)
  - Formula: `update_count / current_step`
- **`.std_dev`** - Standard deviation across episodes (variance measure)
  - Formula: `sqrt(sum((x - mean)²) / n)`
- **`.updates`** - Total number of times this metric was updated in an episode

## Subsections

### General Metrics

**Count:** 1164 metrics

**action.attack.agent:** (8 values / 8 std_devs)
- `env_agent/action.attack.agent`
- `env_agent/action.attack.agent.avg`
- `env_agent/action.attack.agent.avg.std_dev`
- `env_agent/action.attack.agent.first_step`
- `env_agent/action.attack.agent.first_step.std_dev`
- `env_agent/action.attack.agent.last_step`
- `env_agent/action.attack.agent.last_step.std_dev`
- `env_agent/action.attack.agent.max`
- `env_agent/action.attack.agent.max.std_dev`
- `env_agent/action.attack.agent.min`
- `env_agent/action.attack.agent.min.std_dev`
- `env_agent/action.attack.agent.rate`
- `env_agent/action.attack.agent.rate.std_dev`
- `env_agent/action.attack.agent.updates`
- `env_agent/action.attack.agent.updates.std_dev`
- `env_agent/action.attack.agent.std_dev`

**action.attack.agent.agent:** (8 values / 8 std_devs)
- `env_agent/action.attack.agent.agent`
- `env_agent/action.attack.agent.agent.avg`
- `env_agent/action.attack.agent.agent.avg.std_dev`
- `env_agent/action.attack.agent.agent.first_step`
- `env_agent/action.attack.agent.agent.first_step.std_dev`
- `env_agent/action.attack.agent.agent.last_step`
- `env_agent/action.attack.agent.agent.last_step.std_dev`
- `env_agent/action.attack.agent.agent.max`
- `env_agent/action.attack.agent.agent.max.std_dev`
- `env_agent/action.attack.agent.agent.min`
- `env_agent/action.attack.agent.agent.min.std_dev`
- `env_agent/action.attack.agent.agent.rate`
- `env_agent/action.attack.agent.agent.rate.std_dev`
- `env_agent/action.attack.agent.agent.updates`
- `env_agent/action.attack.agent.agent.updates.std_dev`
- `env_agent/action.attack.agent.agent.std_dev`

**action.attack.agent.agent.agent:** (8 values / 8 std_devs)
- `env_agent/action.attack.agent.agent.agent`
- `env_agent/action.attack.agent.agent.agent.avg`
- `env_agent/action.attack.agent.agent.agent.avg.std_dev`
- `env_agent/action.attack.agent.agent.agent.first_step`
- `env_agent/action.attack.agent.agent.agent.first_step.std_dev`
- `env_agent/action.attack.agent.agent.agent.last_step`
- `env_agent/action.attack.agent.agent.agent.last_step.std_dev`
- `env_agent/action.attack.agent.agent.agent.max`
- `env_agent/action.attack.agent.agent.agent.max.std_dev`
- `env_agent/action.attack.agent.agent.agent.min`
- `env_agent/action.attack.agent.agent.agent.min.std_dev`
- `env_agent/action.attack.agent.agent.agent.rate`
- `env_agent/action.attack.agent.agent.agent.rate.std_dev`
- `env_agent/action.attack.agent.agent.agent.updates`
- `env_agent/action.attack.agent.agent.agent.updates.std_dev`
- `env_agent/action.attack.agent.agent.agent.std_dev`

**action.attack.failed:** (9 values / 9 std_devs)
- `env_agent/action.attack.failed`
- `env_agent/action.attack.failed.activity_rate`
- `env_agent/action.attack.failed.activity_rate.std_dev`
- `env_agent/action.attack.failed.avg`
- `env_agent/action.attack.failed.avg.std_dev`
- `env_agent/action.attack.failed.first_step`
- `env_agent/action.attack.failed.first_step.std_dev`
- `env_agent/action.attack.failed.last_step`
- `env_agent/action.attack.failed.last_step.std_dev`
- `env_agent/action.attack.failed.max`
- `env_agent/action.attack.failed.max.std_dev`
- `env_agent/action.attack.failed.min`
- `env_agent/action.attack.failed.min.std_dev`
- `env_agent/action.attack.failed.rate`
- `env_agent/action.attack.failed.rate.std_dev`
- `env_agent/action.attack.failed.updates`
- `env_agent/action.attack.failed.updates.std_dev`
- `env_agent/action.attack.failed.std_dev`

**action.attack.success:** (8 values / 8 std_devs)
- `env_agent/action.attack.success`
- `env_agent/action.attack.success.avg`
- `env_agent/action.attack.success.avg.std_dev`
- `env_agent/action.attack.success.first_step`
- `env_agent/action.attack.success.first_step.std_dev`
- `env_agent/action.attack.success.last_step`
- `env_agent/action.attack.success.last_step.std_dev`
- `env_agent/action.attack.success.max`
- `env_agent/action.attack.success.max.std_dev`
- `env_agent/action.attack.success.min`
- `env_agent/action.attack.success.min.std_dev`
- `env_agent/action.attack.success.rate`
  - Success rate of attack actions when attempted.
  - **Interpretation:** Balance between too aggressive (low success) and too passive (few attempts).

- `env_agent/action.attack.success.rate.std_dev`
  - Success rate of attack actions when attempted. (standard deviation)

- `env_agent/action.attack.success.updates`
- `env_agent/action.attack.success.updates.std_dev`
- `env_agent/action.attack.success.std_dev`

**action.attack_nearest.agent:** (9 values / 9 std_devs)
- `env_agent/action.attack_nearest.agent`
- `env_agent/action.attack_nearest.agent.activity_rate`
- `env_agent/action.attack_nearest.agent.activity_rate.std_dev`
- `env_agent/action.attack_nearest.agent.avg`
- `env_agent/action.attack_nearest.agent.avg.std_dev`
- `env_agent/action.attack_nearest.agent.first_step`
- `env_agent/action.attack_nearest.agent.first_step.std_dev`
- `env_agent/action.attack_nearest.agent.last_step`
- `env_agent/action.attack_nearest.agent.last_step.std_dev`
- `env_agent/action.attack_nearest.agent.max`
- `env_agent/action.attack_nearest.agent.max.std_dev`
- `env_agent/action.attack_nearest.agent.min`
- `env_agent/action.attack_nearest.agent.min.std_dev`
- `env_agent/action.attack_nearest.agent.rate`
- `env_agent/action.attack_nearest.agent.rate.std_dev`
- `env_agent/action.attack_nearest.agent.updates`
- `env_agent/action.attack_nearest.agent.updates.std_dev`
- `env_agent/action.attack_nearest.agent.std_dev`

**action.attack_nearest.agent.agent:** (9 values / 9 std_devs)
- `env_agent/action.attack_nearest.agent.agent`
- `env_agent/action.attack_nearest.agent.agent.activity_rate`
- `env_agent/action.attack_nearest.agent.agent.activity_rate.std_dev`
- `env_agent/action.attack_nearest.agent.agent.avg`
- `env_agent/action.attack_nearest.agent.agent.avg.std_dev`
- `env_agent/action.attack_nearest.agent.agent.first_step`
- `env_agent/action.attack_nearest.agent.agent.first_step.std_dev`
- `env_agent/action.attack_nearest.agent.agent.last_step`
- `env_agent/action.attack_nearest.agent.agent.last_step.std_dev`
- `env_agent/action.attack_nearest.agent.agent.max`
- `env_agent/action.attack_nearest.agent.agent.max.std_dev`
- `env_agent/action.attack_nearest.agent.agent.min`
- `env_agent/action.attack_nearest.agent.agent.min.std_dev`
- `env_agent/action.attack_nearest.agent.agent.rate`
- `env_agent/action.attack_nearest.agent.agent.rate.std_dev`
- `env_agent/action.attack_nearest.agent.agent.updates`
- `env_agent/action.attack_nearest.agent.agent.updates.std_dev`
- `env_agent/action.attack_nearest.agent.agent.std_dev`

**action.attack_nearest.agent.agent.agent:** (9 values / 9 std_devs)
- `env_agent/action.attack_nearest.agent.agent.agent`
- `env_agent/action.attack_nearest.agent.agent.agent.activity_rate`
- `env_agent/action.attack_nearest.agent.agent.agent.activity_rate.std_dev`
- `env_agent/action.attack_nearest.agent.agent.agent.avg`
- `env_agent/action.attack_nearest.agent.agent.agent.avg.std_dev`
- `env_agent/action.attack_nearest.agent.agent.agent.first_step`
- `env_agent/action.attack_nearest.agent.agent.agent.first_step.std_dev`
- `env_agent/action.attack_nearest.agent.agent.agent.last_step`
- `env_agent/action.attack_nearest.agent.agent.agent.last_step.std_dev`
- `env_agent/action.attack_nearest.agent.agent.agent.max`
- `env_agent/action.attack_nearest.agent.agent.agent.max.std_dev`
- `env_agent/action.attack_nearest.agent.agent.agent.min`
- `env_agent/action.attack_nearest.agent.agent.agent.min.std_dev`
- `env_agent/action.attack_nearest.agent.agent.agent.rate`
- `env_agent/action.attack_nearest.agent.agent.agent.rate.std_dev`
- `env_agent/action.attack_nearest.agent.agent.agent.updates`
- `env_agent/action.attack_nearest.agent.agent.agent.updates.std_dev`
- `env_agent/action.attack_nearest.agent.agent.agent.std_dev`

**action.attack_nearest.failed:** (9 values / 9 std_devs)
- `env_agent/action.attack_nearest.failed`
- `env_agent/action.attack_nearest.failed.activity_rate`
- `env_agent/action.attack_nearest.failed.activity_rate.std_dev`
- `env_agent/action.attack_nearest.failed.avg`
- `env_agent/action.attack_nearest.failed.avg.std_dev`
- `env_agent/action.attack_nearest.failed.first_step`
- `env_agent/action.attack_nearest.failed.first_step.std_dev`
- `env_agent/action.attack_nearest.failed.last_step`
- `env_agent/action.attack_nearest.failed.last_step.std_dev`
- `env_agent/action.attack_nearest.failed.max`
- `env_agent/action.attack_nearest.failed.max.std_dev`
- `env_agent/action.attack_nearest.failed.min`
- `env_agent/action.attack_nearest.failed.min.std_dev`
- `env_agent/action.attack_nearest.failed.rate`
- `env_agent/action.attack_nearest.failed.rate.std_dev`
- `env_agent/action.attack_nearest.failed.updates`
- `env_agent/action.attack_nearest.failed.updates.std_dev`
- `env_agent/action.attack_nearest.failed.std_dev`

**action.attack_nearest.success:** (8 values / 8 std_devs)
- `env_agent/action.attack_nearest.success`
- `env_agent/action.attack_nearest.success.avg`
- `env_agent/action.attack_nearest.success.avg.std_dev`
- `env_agent/action.attack_nearest.success.first_step`
- `env_agent/action.attack_nearest.success.first_step.std_dev`
- `env_agent/action.attack_nearest.success.last_step`
- `env_agent/action.attack_nearest.success.last_step.std_dev`
- `env_agent/action.attack_nearest.success.max`
- `env_agent/action.attack_nearest.success.max.std_dev`
- `env_agent/action.attack_nearest.success.min`
- `env_agent/action.attack_nearest.success.min.std_dev`
- `env_agent/action.attack_nearest.success.rate`
- `env_agent/action.attack_nearest.success.rate.std_dev`
- `env_agent/action.attack_nearest.success.updates`
- `env_agent/action.attack_nearest.success.updates.std_dev`
- `env_agent/action.attack_nearest.success.std_dev`

**action.change_color.success:** (9 values / 9 std_devs)
- `env_agent/action.change_color.success`
- `env_agent/action.change_color.success.activity_rate`
- `env_agent/action.change_color.success.activity_rate.std_dev`
- `env_agent/action.change_color.success.avg`
- `env_agent/action.change_color.success.avg.std_dev`
- `env_agent/action.change_color.success.first_step`
- `env_agent/action.change_color.success.first_step.std_dev`
- `env_agent/action.change_color.success.last_step`
- `env_agent/action.change_color.success.last_step.std_dev`
- `env_agent/action.change_color.success.max`
- `env_agent/action.change_color.success.max.std_dev`
- `env_agent/action.change_color.success.min`
- `env_agent/action.change_color.success.min.std_dev`
- `env_agent/action.change_color.success.rate`
- `env_agent/action.change_color.success.rate.std_dev`
- `env_agent/action.change_color.success.updates`
- `env_agent/action.change_color.success.updates.std_dev`
- `env_agent/action.change_color.success.std_dev`

**action.failure_penalty:** (9 values / 9 std_devs)
- `env_agent/action.failure_penalty`
  - Penalty applied when agents attempt invalid actions.
  - **Interpretation:** High values indicate agents haven't learned action preconditions.

- `env_agent/action.failure_penalty.activity_rate`
  - Penalty applied when agents attempt invalid actions. (fraction of steps where this was active)

- `env_agent/action.failure_penalty.activity_rate.std_dev`
  - Penalty applied when agents attempt invalid actions. (fraction of steps where this was active) (standard deviation)

- `env_agent/action.failure_penalty.avg`
  - Penalty applied when agents attempt invalid actions. (average value)

- `env_agent/action.failure_penalty.avg.std_dev`
  - Penalty applied when agents attempt invalid actions. (average value) (standard deviation)

- `env_agent/action.failure_penalty.first_step`
  - Penalty applied when agents attempt invalid actions. (first step where this occurred)

- `env_agent/action.failure_penalty.first_step.std_dev`
  - Penalty applied when agents attempt invalid actions. (first step where this occurred) (standard deviation)

- `env_agent/action.failure_penalty.last_step`
  - Penalty applied when agents attempt invalid actions. (last step where this occurred)

- `env_agent/action.failure_penalty.last_step.std_dev`
  - Penalty applied when agents attempt invalid actions. (last step where this occurred) (standard deviation)

- `env_agent/action.failure_penalty.max`
  - Penalty applied when agents attempt invalid actions. (maximum value)

- `env_agent/action.failure_penalty.max.std_dev`
  - Penalty applied when agents attempt invalid actions. (maximum value) (standard deviation)

- `env_agent/action.failure_penalty.min`
  - Penalty applied when agents attempt invalid actions. (minimum value)

- `env_agent/action.failure_penalty.min.std_dev`
  - Penalty applied when agents attempt invalid actions. (minimum value) (standard deviation)

- `env_agent/action.failure_penalty.rate`
  - Penalty applied when agents attempt invalid actions. (occurrences per step)

- `env_agent/action.failure_penalty.rate.std_dev`
  - Penalty applied when agents attempt invalid actions. (occurrences per step) (standard deviation)

- `env_agent/action.failure_penalty.updates`
- `env_agent/action.failure_penalty.updates.std_dev`
- `env_agent/action.failure_penalty.std_dev`
  - Penalty applied when agents attempt invalid actions. (standard deviation)


**action.get_output.failed:** (9 values / 9 std_devs)
- `env_agent/action.get_output.failed`
- `env_agent/action.get_output.failed.activity_rate`
- `env_agent/action.get_output.failed.activity_rate.std_dev`
- `env_agent/action.get_output.failed.avg`
- `env_agent/action.get_output.failed.avg.std_dev`
- `env_agent/action.get_output.failed.first_step`
- `env_agent/action.get_output.failed.first_step.std_dev`
- `env_agent/action.get_output.failed.last_step`
- `env_agent/action.get_output.failed.last_step.std_dev`
- `env_agent/action.get_output.failed.max`
- `env_agent/action.get_output.failed.max.std_dev`
- `env_agent/action.get_output.failed.min`
- `env_agent/action.get_output.failed.min.std_dev`
- `env_agent/action.get_output.failed.rate`
- `env_agent/action.get_output.failed.rate.std_dev`
- `env_agent/action.get_output.failed.updates`
- `env_agent/action.get_output.failed.updates.std_dev`
- `env_agent/action.get_output.failed.std_dev`

**action.get_output.success:** (9 values / 9 std_devs)
- `env_agent/action.get_output.success`
- `env_agent/action.get_output.success.activity_rate`
- `env_agent/action.get_output.success.activity_rate.std_dev`
- `env_agent/action.get_output.success.avg`
- `env_agent/action.get_output.success.avg.std_dev`
- `env_agent/action.get_output.success.first_step`
- `env_agent/action.get_output.success.first_step.std_dev`
- `env_agent/action.get_output.success.last_step`
- `env_agent/action.get_output.success.last_step.std_dev`
- `env_agent/action.get_output.success.max`
- `env_agent/action.get_output.success.max.std_dev`
- `env_agent/action.get_output.success.min`
- `env_agent/action.get_output.success.min.std_dev`
- `env_agent/action.get_output.success.rate`
- `env_agent/action.get_output.success.rate.std_dev`
- `env_agent/action.get_output.success.updates`
- `env_agent/action.get_output.success.updates.std_dev`
- `env_agent/action.get_output.success.std_dev`

**action.move.failed:** (9 values / 9 std_devs)
- `env_agent/action.move.failed`
- `env_agent/action.move.failed.activity_rate`
- `env_agent/action.move.failed.activity_rate.std_dev`
- `env_agent/action.move.failed.avg`
- `env_agent/action.move.failed.avg.std_dev`
- `env_agent/action.move.failed.first_step`
- `env_agent/action.move.failed.first_step.std_dev`
- `env_agent/action.move.failed.last_step`
- `env_agent/action.move.failed.last_step.std_dev`
- `env_agent/action.move.failed.max`
- `env_agent/action.move.failed.max.std_dev`
- `env_agent/action.move.failed.min`
- `env_agent/action.move.failed.min.std_dev`
- `env_agent/action.move.failed.rate`
- `env_agent/action.move.failed.rate.std_dev`
- `env_agent/action.move.failed.updates`
- `env_agent/action.move.failed.updates.std_dev`
- `env_agent/action.move.failed.std_dev`

**action.move.success:** (9 values / 9 std_devs)
- `env_agent/action.move.success`
- `env_agent/action.move.success.activity_rate`
- `env_agent/action.move.success.activity_rate.std_dev`
- `env_agent/action.move.success.avg`
- `env_agent/action.move.success.avg.std_dev`
- `env_agent/action.move.success.first_step`
- `env_agent/action.move.success.first_step.std_dev`
- `env_agent/action.move.success.last_step`
- `env_agent/action.move.success.last_step.std_dev`
- `env_agent/action.move.success.max`
- `env_agent/action.move.success.max.std_dev`
- `env_agent/action.move.success.min`
- `env_agent/action.move.success.min.std_dev`
- `env_agent/action.move.success.rate`
  - Success rate of agent movement actions.
  - **Interpretation:** Low rates may indicate crowded environments or poor navigation policy.

- `env_agent/action.move.success.rate.std_dev`
  - Success rate of agent movement actions. (standard deviation)

- `env_agent/action.move.success.updates`
- `env_agent/action.move.success.updates.std_dev`
- `env_agent/action.move.success.std_dev`

**action.noop.success:** (9 values / 9 std_devs)
- `env_agent/action.noop.success`
- `env_agent/action.noop.success.activity_rate`
- `env_agent/action.noop.success.activity_rate.std_dev`
- `env_agent/action.noop.success.avg`
- `env_agent/action.noop.success.avg.std_dev`
- `env_agent/action.noop.success.first_step`
- `env_agent/action.noop.success.first_step.std_dev`
- `env_agent/action.noop.success.last_step`
- `env_agent/action.noop.success.last_step.std_dev`
- `env_agent/action.noop.success.max`
- `env_agent/action.noop.success.max.std_dev`
- `env_agent/action.noop.success.min`
- `env_agent/action.noop.success.min.std_dev`
- `env_agent/action.noop.success.rate`
- `env_agent/action.noop.success.rate.std_dev`
- `env_agent/action.noop.success.updates`
- `env_agent/action.noop.success.updates.std_dev`
- `env_agent/action.noop.success.std_dev`

**action.put_recipe_items.failed:** (9 values / 9 std_devs)
- `env_agent/action.put_recipe_items.failed`
- `env_agent/action.put_recipe_items.failed.activity_rate`
- `env_agent/action.put_recipe_items.failed.activity_rate.std_dev`
- `env_agent/action.put_recipe_items.failed.avg`
- `env_agent/action.put_recipe_items.failed.avg.std_dev`
- `env_agent/action.put_recipe_items.failed.first_step`
- `env_agent/action.put_recipe_items.failed.first_step.std_dev`
- `env_agent/action.put_recipe_items.failed.last_step`
- `env_agent/action.put_recipe_items.failed.last_step.std_dev`
- `env_agent/action.put_recipe_items.failed.max`
- `env_agent/action.put_recipe_items.failed.max.std_dev`
- `env_agent/action.put_recipe_items.failed.min`
- `env_agent/action.put_recipe_items.failed.min.std_dev`
- `env_agent/action.put_recipe_items.failed.rate`
- `env_agent/action.put_recipe_items.failed.rate.std_dev`
- `env_agent/action.put_recipe_items.failed.updates`
- `env_agent/action.put_recipe_items.failed.updates.std_dev`
- `env_agent/action.put_recipe_items.failed.std_dev`

**action.put_recipe_items.success:** (9 values / 9 std_devs)
- `env_agent/action.put_recipe_items.success`
- `env_agent/action.put_recipe_items.success.activity_rate`
- `env_agent/action.put_recipe_items.success.activity_rate.std_dev`
- `env_agent/action.put_recipe_items.success.avg`
- `env_agent/action.put_recipe_items.success.avg.std_dev`
- `env_agent/action.put_recipe_items.success.first_step`
- `env_agent/action.put_recipe_items.success.first_step.std_dev`
- `env_agent/action.put_recipe_items.success.last_step`
- `env_agent/action.put_recipe_items.success.last_step.std_dev`
- `env_agent/action.put_recipe_items.success.max`
- `env_agent/action.put_recipe_items.success.max.std_dev`
- `env_agent/action.put_recipe_items.success.min`
- `env_agent/action.put_recipe_items.success.min.std_dev`
- `env_agent/action.put_recipe_items.success.rate`
- `env_agent/action.put_recipe_items.success.rate.std_dev`
- `env_agent/action.put_recipe_items.success.updates`
- `env_agent/action.put_recipe_items.success.updates.std_dev`
- `env_agent/action.put_recipe_items.success.std_dev`

**action.rotate.success:** (9 values / 9 std_devs)
- `env_agent/action.rotate.success`
- `env_agent/action.rotate.success.activity_rate`
- `env_agent/action.rotate.success.activity_rate.std_dev`
- `env_agent/action.rotate.success.avg`
- `env_agent/action.rotate.success.avg.std_dev`
- `env_agent/action.rotate.success.first_step`
- `env_agent/action.rotate.success.first_step.std_dev`
- `env_agent/action.rotate.success.last_step`
- `env_agent/action.rotate.success.last_step.std_dev`
- `env_agent/action.rotate.success.max`
- `env_agent/action.rotate.success.max.std_dev`
- `env_agent/action.rotate.success.min`
- `env_agent/action.rotate.success.min.std_dev`
- `env_agent/action.rotate.success.rate`
- `env_agent/action.rotate.success.rate.std_dev`
- `env_agent/action.rotate.success.updates`
- `env_agent/action.rotate.success.updates.std_dev`
- `env_agent/action.rotate.success.std_dev`

**action.swap.agent:** (8 values / 8 std_devs)
- `env_agent/action.swap.agent`
- `env_agent/action.swap.agent.avg`
- `env_agent/action.swap.agent.avg.std_dev`
- `env_agent/action.swap.agent.first_step`
- `env_agent/action.swap.agent.first_step.std_dev`
- `env_agent/action.swap.agent.last_step`
- `env_agent/action.swap.agent.last_step.std_dev`
- `env_agent/action.swap.agent.max`
- `env_agent/action.swap.agent.max.std_dev`
- `env_agent/action.swap.agent.min`
- `env_agent/action.swap.agent.min.std_dev`
- `env_agent/action.swap.agent.rate`
- `env_agent/action.swap.agent.rate.std_dev`
- `env_agent/action.swap.agent.updates`
- `env_agent/action.swap.agent.updates.std_dev`
- `env_agent/action.swap.agent.std_dev`

**action.swap.block:** (9 values / 9 std_devs)
- `env_agent/action.swap.block`
- `env_agent/action.swap.block.activity_rate`
- `env_agent/action.swap.block.activity_rate.std_dev`
- `env_agent/action.swap.block.avg`
- `env_agent/action.swap.block.avg.std_dev`
- `env_agent/action.swap.block.first_step`
- `env_agent/action.swap.block.first_step.std_dev`
- `env_agent/action.swap.block.last_step`
- `env_agent/action.swap.block.last_step.std_dev`
- `env_agent/action.swap.block.max`
- `env_agent/action.swap.block.max.std_dev`
- `env_agent/action.swap.block.min`
- `env_agent/action.swap.block.min.std_dev`
- `env_agent/action.swap.block.rate`
- `env_agent/action.swap.block.rate.std_dev`
- `env_agent/action.swap.block.updates`
- `env_agent/action.swap.block.updates.std_dev`
- `env_agent/action.swap.block.std_dev`

**action.swap.failed:** (9 values / 9 std_devs)
- `env_agent/action.swap.failed`
- `env_agent/action.swap.failed.activity_rate`
- `env_agent/action.swap.failed.activity_rate.std_dev`
- `env_agent/action.swap.failed.avg`
- `env_agent/action.swap.failed.avg.std_dev`
- `env_agent/action.swap.failed.first_step`
- `env_agent/action.swap.failed.first_step.std_dev`
- `env_agent/action.swap.failed.last_step`
- `env_agent/action.swap.failed.last_step.std_dev`
- `env_agent/action.swap.failed.max`
- `env_agent/action.swap.failed.max.std_dev`
- `env_agent/action.swap.failed.min`
- `env_agent/action.swap.failed.min.std_dev`
- `env_agent/action.swap.failed.rate`
- `env_agent/action.swap.failed.rate.std_dev`
- `env_agent/action.swap.failed.updates`
- `env_agent/action.swap.failed.updates.std_dev`
- `env_agent/action.swap.failed.std_dev`

**action.swap.success:** (9 values / 9 std_devs)
- `env_agent/action.swap.success`
- `env_agent/action.swap.success.activity_rate`
- `env_agent/action.swap.success.activity_rate.std_dev`
- `env_agent/action.swap.success.avg`
- `env_agent/action.swap.success.avg.std_dev`
- `env_agent/action.swap.success.first_step`
- `env_agent/action.swap.success.first_step.std_dev`
- `env_agent/action.swap.success.last_step`
- `env_agent/action.swap.success.last_step.std_dev`
- `env_agent/action.swap.success.max`
- `env_agent/action.swap.success.max.std_dev`
- `env_agent/action.swap.success.min`
- `env_agent/action.swap.success.min.std_dev`
- `env_agent/action.swap.success.rate`
- `env_agent/action.swap.success.rate.std_dev`
- `env_agent/action.swap.success.updates`
- `env_agent/action.swap.success.updates.std_dev`
- `env_agent/action.swap.success.std_dev`

**armor.gained:** (9 values / 9 std_devs)
- `env_agent/armor.gained`
- `env_agent/armor.gained.activity_rate`
- `env_agent/armor.gained.activity_rate.std_dev`
- `env_agent/armor.gained.avg`
- `env_agent/armor.gained.avg.std_dev`
- `env_agent/armor.gained.first_step`
- `env_agent/armor.gained.first_step.std_dev`
- `env_agent/armor.gained.last_step`
- `env_agent/armor.gained.last_step.std_dev`
- `env_agent/armor.gained.max`
- `env_agent/armor.gained.max.std_dev`
- `env_agent/armor.gained.min`
- `env_agent/armor.gained.min.std_dev`
- `env_agent/armor.gained.rate`
- `env_agent/armor.gained.rate.std_dev`
- `env_agent/armor.gained.updates`
- `env_agent/armor.gained.updates.std_dev`
- `env_agent/armor.gained.std_dev`

**armor.get:** (9 values / 9 std_devs)
- `env_agent/armor.get`
- `env_agent/armor.get.activity_rate`
- `env_agent/armor.get.activity_rate.std_dev`
- `env_agent/armor.get.avg`
- `env_agent/armor.get.avg.std_dev`
- `env_agent/armor.get.first_step`
- `env_agent/armor.get.first_step.std_dev`
- `env_agent/armor.get.last_step`
- `env_agent/armor.get.last_step.std_dev`
- `env_agent/armor.get.max`
- `env_agent/armor.get.max.std_dev`
- `env_agent/armor.get.min`
- `env_agent/armor.get.min.std_dev`
- `env_agent/armor.get.rate`
- `env_agent/armor.get.rate.std_dev`
- `env_agent/armor.get.updates`
- `env_agent/armor.get.updates.std_dev`
- `env_agent/armor.get.std_dev`

**armor.lost:** (8 values / 8 std_devs)
- `env_agent/armor.lost`
- `env_agent/armor.lost.avg`
- `env_agent/armor.lost.avg.std_dev`
- `env_agent/armor.lost.first_step`
- `env_agent/armor.lost.first_step.std_dev`
- `env_agent/armor.lost.last_step`
- `env_agent/armor.lost.last_step.std_dev`
- `env_agent/armor.lost.max`
- `env_agent/armor.lost.max.std_dev`
- `env_agent/armor.lost.min`
- `env_agent/armor.lost.min.std_dev`
- `env_agent/armor.lost.rate`
- `env_agent/armor.lost.rate.std_dev`
- `env_agent/armor.lost.updates`
- `env_agent/armor.lost.updates.std_dev`
- `env_agent/armor.lost.std_dev`

**attack.blocked.agent:** (8 values / 8 std_devs)
- `env_agent/attack.blocked.agent`
- `env_agent/attack.blocked.agent.avg`
- `env_agent/attack.blocked.agent.avg.std_dev`
- `env_agent/attack.blocked.agent.first_step`
- `env_agent/attack.blocked.agent.first_step.std_dev`
- `env_agent/attack.blocked.agent.last_step`
- `env_agent/attack.blocked.agent.last_step.std_dev`
- `env_agent/attack.blocked.agent.max`
- `env_agent/attack.blocked.agent.max.std_dev`
- `env_agent/attack.blocked.agent.min`
- `env_agent/attack.blocked.agent.min.std_dev`
- `env_agent/attack.blocked.agent.rate`
- `env_agent/attack.blocked.agent.rate.std_dev`
- `env_agent/attack.blocked.agent.updates`
- `env_agent/attack.blocked.agent.updates.std_dev`
- `env_agent/attack.blocked.agent.std_dev`

**attack.blocked.agent.agent:** (8 values / 8 std_devs)
- `env_agent/attack.blocked.agent.agent`
- `env_agent/attack.blocked.agent.agent.avg`
- `env_agent/attack.blocked.agent.agent.avg.std_dev`
- `env_agent/attack.blocked.agent.agent.first_step`
- `env_agent/attack.blocked.agent.agent.first_step.std_dev`
- `env_agent/attack.blocked.agent.agent.last_step`
- `env_agent/attack.blocked.agent.agent.last_step.std_dev`
- `env_agent/attack.blocked.agent.agent.max`
- `env_agent/attack.blocked.agent.agent.max.std_dev`
- `env_agent/attack.blocked.agent.agent.min`
- `env_agent/attack.blocked.agent.agent.min.std_dev`
- `env_agent/attack.blocked.agent.agent.rate`
- `env_agent/attack.blocked.agent.agent.rate.std_dev`
- `env_agent/attack.blocked.agent.agent.updates`
- `env_agent/attack.blocked.agent.agent.updates.std_dev`
- `env_agent/attack.blocked.agent.agent.std_dev`

**attack.loss.agent:** (9 values / 9 std_devs)
- `env_agent/attack.loss.agent`
- `env_agent/attack.loss.agent.activity_rate`
- `env_agent/attack.loss.agent.activity_rate.std_dev`
- `env_agent/attack.loss.agent.avg`
- `env_agent/attack.loss.agent.avg.std_dev`
- `env_agent/attack.loss.agent.first_step`
- `env_agent/attack.loss.agent.first_step.std_dev`
- `env_agent/attack.loss.agent.last_step`
- `env_agent/attack.loss.agent.last_step.std_dev`
- `env_agent/attack.loss.agent.max`
- `env_agent/attack.loss.agent.max.std_dev`
- `env_agent/attack.loss.agent.min`
- `env_agent/attack.loss.agent.min.std_dev`
- `env_agent/attack.loss.agent.rate`
- `env_agent/attack.loss.agent.rate.std_dev`
- `env_agent/attack.loss.agent.updates`
- `env_agent/attack.loss.agent.updates.std_dev`
- `env_agent/attack.loss.agent.std_dev`

**attack.loss.agent.agent:** (9 values / 9 std_devs)
- `env_agent/attack.loss.agent.agent`
- `env_agent/attack.loss.agent.agent.activity_rate`
- `env_agent/attack.loss.agent.agent.activity_rate.std_dev`
- `env_agent/attack.loss.agent.agent.avg`
- `env_agent/attack.loss.agent.agent.avg.std_dev`
- `env_agent/attack.loss.agent.agent.first_step`
- `env_agent/attack.loss.agent.agent.first_step.std_dev`
- `env_agent/attack.loss.agent.agent.last_step`
- `env_agent/attack.loss.agent.agent.last_step.std_dev`
- `env_agent/attack.loss.agent.agent.max`
- `env_agent/attack.loss.agent.agent.max.std_dev`
- `env_agent/attack.loss.agent.agent.min`
- `env_agent/attack.loss.agent.agent.min.std_dev`
- `env_agent/attack.loss.agent.agent.rate`
- `env_agent/attack.loss.agent.agent.rate.std_dev`
- `env_agent/attack.loss.agent.agent.updates`
- `env_agent/attack.loss.agent.agent.updates.std_dev`
- `env_agent/attack.loss.agent.agent.std_dev`

**attack.loss.from_own_team.agent:** (9 values / 9 std_devs)
- `env_agent/attack.loss.from_own_team.agent`
- `env_agent/attack.loss.from_own_team.agent.activity_rate`
- `env_agent/attack.loss.from_own_team.agent.activity_rate.std_dev`
- `env_agent/attack.loss.from_own_team.agent.avg`
- `env_agent/attack.loss.from_own_team.agent.avg.std_dev`
- `env_agent/attack.loss.from_own_team.agent.first_step`
- `env_agent/attack.loss.from_own_team.agent.first_step.std_dev`
- `env_agent/attack.loss.from_own_team.agent.last_step`
- `env_agent/attack.loss.from_own_team.agent.last_step.std_dev`
- `env_agent/attack.loss.from_own_team.agent.max`
- `env_agent/attack.loss.from_own_team.agent.max.std_dev`
- `env_agent/attack.loss.from_own_team.agent.min`
- `env_agent/attack.loss.from_own_team.agent.min.std_dev`
- `env_agent/attack.loss.from_own_team.agent.rate`
- `env_agent/attack.loss.from_own_team.agent.rate.std_dev`
- `env_agent/attack.loss.from_own_team.agent.updates`
- `env_agent/attack.loss.from_own_team.agent.updates.std_dev`
- `env_agent/attack.loss.from_own_team.agent.std_dev`

**attack.own_team.agent:** (9 values / 9 std_devs)
- `env_agent/attack.own_team.agent`
- `env_agent/attack.own_team.agent.activity_rate`
- `env_agent/attack.own_team.agent.activity_rate.std_dev`
- `env_agent/attack.own_team.agent.avg`
- `env_agent/attack.own_team.agent.avg.std_dev`
- `env_agent/attack.own_team.agent.first_step`
- `env_agent/attack.own_team.agent.first_step.std_dev`
- `env_agent/attack.own_team.agent.last_step`
- `env_agent/attack.own_team.agent.last_step.std_dev`
- `env_agent/attack.own_team.agent.max`
- `env_agent/attack.own_team.agent.max.std_dev`
- `env_agent/attack.own_team.agent.min`
- `env_agent/attack.own_team.agent.min.std_dev`
- `env_agent/attack.own_team.agent.rate`
- `env_agent/attack.own_team.agent.rate.std_dev`
- `env_agent/attack.own_team.agent.updates`
- `env_agent/attack.own_team.agent.updates.std_dev`
- `env_agent/attack.own_team.agent.std_dev`

**attack.win.agent:** (9 values / 9 std_devs)
- `env_agent/attack.win.agent`
- `env_agent/attack.win.agent.activity_rate`
- `env_agent/attack.win.agent.activity_rate.std_dev`
- `env_agent/attack.win.agent.avg`
- `env_agent/attack.win.agent.avg.std_dev`
- `env_agent/attack.win.agent.first_step`
- `env_agent/attack.win.agent.first_step.std_dev`
- `env_agent/attack.win.agent.last_step`
- `env_agent/attack.win.agent.last_step.std_dev`
- `env_agent/attack.win.agent.max`
- `env_agent/attack.win.agent.max.std_dev`
- `env_agent/attack.win.agent.min`
- `env_agent/attack.win.agent.min.std_dev`
- `env_agent/attack.win.agent.rate`
- `env_agent/attack.win.agent.rate.std_dev`
- `env_agent/attack.win.agent.updates`
- `env_agent/attack.win.agent.updates.std_dev`
- `env_agent/attack.win.agent.std_dev`

**attack.win.agent.agent:** (9 values / 9 std_devs)
- `env_agent/attack.win.agent.agent`
- `env_agent/attack.win.agent.agent.activity_rate`
- `env_agent/attack.win.agent.agent.activity_rate.std_dev`
- `env_agent/attack.win.agent.agent.avg`
- `env_agent/attack.win.agent.agent.avg.std_dev`
- `env_agent/attack.win.agent.agent.first_step`
- `env_agent/attack.win.agent.agent.first_step.std_dev`
- `env_agent/attack.win.agent.agent.last_step`
- `env_agent/attack.win.agent.agent.last_step.std_dev`
- `env_agent/attack.win.agent.agent.max`
- `env_agent/attack.win.agent.agent.max.std_dev`
- `env_agent/attack.win.agent.agent.min`
- `env_agent/attack.win.agent.agent.min.std_dev`
- `env_agent/attack.win.agent.agent.rate`
- `env_agent/attack.win.agent.agent.rate.std_dev`
- `env_agent/attack.win.agent.agent.updates`
- `env_agent/attack.win.agent.agent.updates.std_dev`
- `env_agent/attack.win.agent.agent.std_dev`

**attack.win.own_team.agent:** (9 values / 9 std_devs)
- `env_agent/attack.win.own_team.agent`
- `env_agent/attack.win.own_team.agent.activity_rate`
- `env_agent/attack.win.own_team.agent.activity_rate.std_dev`
- `env_agent/attack.win.own_team.agent.avg`
- `env_agent/attack.win.own_team.agent.avg.std_dev`
- `env_agent/attack.win.own_team.agent.first_step`
- `env_agent/attack.win.own_team.agent.first_step.std_dev`
- `env_agent/attack.win.own_team.agent.last_step`
- `env_agent/attack.win.own_team.agent.last_step.std_dev`
- `env_agent/attack.win.own_team.agent.max`
- `env_agent/attack.win.own_team.agent.max.std_dev`
- `env_agent/attack.win.own_team.agent.min`
- `env_agent/attack.win.own_team.agent.min.std_dev`
- `env_agent/attack.win.own_team.agent.rate`
- `env_agent/attack.win.own_team.agent.rate.std_dev`
- `env_agent/attack.win.own_team.agent.updates`
- `env_agent/attack.win.own_team.agent.updates.std_dev`
- `env_agent/attack.win.own_team.agent.std_dev`

**battery.red.gained:** (9 values / 9 std_devs)
- `env_agent/battery.red.gained`
- `env_agent/battery.red.gained.activity_rate`
- `env_agent/battery.red.gained.activity_rate.std_dev`
- `env_agent/battery.red.gained.avg`
- `env_agent/battery.red.gained.avg.std_dev`
- `env_agent/battery.red.gained.first_step`
- `env_agent/battery.red.gained.first_step.std_dev`
- `env_agent/battery.red.gained.last_step`
- `env_agent/battery.red.gained.last_step.std_dev`
- `env_agent/battery.red.gained.max`
- `env_agent/battery.red.gained.max.std_dev`
- `env_agent/battery.red.gained.min`
- `env_agent/battery.red.gained.min.std_dev`
- `env_agent/battery.red.gained.rate`
- `env_agent/battery.red.gained.rate.std_dev`
- `env_agent/battery.red.gained.updates`
- `env_agent/battery.red.gained.updates.std_dev`
- `env_agent/battery.red.gained.std_dev`

**battery.red.get:** (9 values / 9 std_devs)
- `env_agent/battery.red.get`
- `env_agent/battery.red.get.activity_rate`
- `env_agent/battery.red.get.activity_rate.std_dev`
- `env_agent/battery.red.get.avg`
- `env_agent/battery.red.get.avg.std_dev`
- `env_agent/battery.red.get.first_step`
- `env_agent/battery.red.get.first_step.std_dev`
- `env_agent/battery.red.get.last_step`
- `env_agent/battery.red.get.last_step.std_dev`
- `env_agent/battery.red.get.max`
- `env_agent/battery.red.get.max.std_dev`
- `env_agent/battery.red.get.min`
- `env_agent/battery.red.get.min.std_dev`
- `env_agent/battery.red.get.rate`
- `env_agent/battery.red.get.rate.std_dev`
- `env_agent/battery.red.get.updates`
- `env_agent/battery.red.get.updates.std_dev`
- `env_agent/battery.red.get.std_dev`

**battery.red.lost:** (9 values / 9 std_devs)
- `env_agent/battery.red.lost`
- `env_agent/battery.red.lost.activity_rate`
- `env_agent/battery.red.lost.activity_rate.std_dev`
- `env_agent/battery.red.lost.avg`
- `env_agent/battery.red.lost.avg.std_dev`
- `env_agent/battery.red.lost.first_step`
- `env_agent/battery.red.lost.first_step.std_dev`
- `env_agent/battery.red.lost.last_step`
- `env_agent/battery.red.lost.last_step.std_dev`
- `env_agent/battery.red.lost.max`
- `env_agent/battery.red.lost.max.std_dev`
- `env_agent/battery.red.lost.min`
- `env_agent/battery.red.lost.min.std_dev`
- `env_agent/battery.red.lost.rate`
- `env_agent/battery.red.lost.rate.std_dev`
- `env_agent/battery.red.lost.updates`
- `env_agent/battery.red.lost.updates.std_dev`
- `env_agent/battery.red.lost.std_dev`

**battery.red.put:** (9 values / 9 std_devs)
- `env_agent/battery.red.put`
- `env_agent/battery.red.put.activity_rate`
- `env_agent/battery.red.put.activity_rate.std_dev`
- `env_agent/battery.red.put.avg`
- `env_agent/battery.red.put.avg.std_dev`
- `env_agent/battery.red.put.first_step`
- `env_agent/battery.red.put.first_step.std_dev`
- `env_agent/battery.red.put.last_step`
- `env_agent/battery.red.put.last_step.std_dev`
- `env_agent/battery.red.put.max`
- `env_agent/battery.red.put.max.std_dev`
- `env_agent/battery.red.put.min`
- `env_agent/battery.red.put.min.std_dev`
- `env_agent/battery.red.put.rate`
- `env_agent/battery.red.put.rate.std_dev`
- `env_agent/battery.red.put.updates`
- `env_agent/battery.red.put.updates.std_dev`
- `env_agent/battery.red.put.std_dev`

**battery.red.stolen.agent:** (8 values / 8 std_devs)
- `env_agent/battery.red.stolen.agent`
- `env_agent/battery.red.stolen.agent.avg`
- `env_agent/battery.red.stolen.agent.avg.std_dev`
- `env_agent/battery.red.stolen.agent.first_step`
- `env_agent/battery.red.stolen.agent.first_step.std_dev`
- `env_agent/battery.red.stolen.agent.last_step`
- `env_agent/battery.red.stolen.agent.last_step.std_dev`
- `env_agent/battery.red.stolen.agent.max`
- `env_agent/battery.red.stolen.agent.max.std_dev`
- `env_agent/battery.red.stolen.agent.min`
- `env_agent/battery.red.stolen.agent.min.std_dev`
- `env_agent/battery.red.stolen.agent.rate`
- `env_agent/battery.red.stolen.agent.rate.std_dev`
- `env_agent/battery.red.stolen.agent.updates`
- `env_agent/battery.red.stolen.agent.updates.std_dev`
- `env_agent/battery.red.stolen.agent.std_dev`

**battery.red.stolen_from.agent:** (8 values / 8 std_devs)
- `env_agent/battery.red.stolen_from.agent`
- `env_agent/battery.red.stolen_from.agent.avg`
- `env_agent/battery.red.stolen_from.agent.avg.std_dev`
- `env_agent/battery.red.stolen_from.agent.first_step`
- `env_agent/battery.red.stolen_from.agent.first_step.std_dev`
- `env_agent/battery.red.stolen_from.agent.last_step`
- `env_agent/battery.red.stolen_from.agent.last_step.std_dev`
- `env_agent/battery.red.stolen_from.agent.max`
- `env_agent/battery.red.stolen_from.agent.max.std_dev`
- `env_agent/battery.red.stolen_from.agent.min`
- `env_agent/battery.red.stolen_from.agent.min.std_dev`
- `env_agent/battery.red.stolen_from.agent.rate`
- `env_agent/battery.red.stolen_from.agent.rate.std_dev`
- `env_agent/battery.red.stolen_from.agent.updates`
- `env_agent/battery.red.stolen_from.agent.updates.std_dev`
- `env_agent/battery.red.stolen_from.agent.std_dev`

**blueprint.gained:** (9 values / 9 std_devs)
- `env_agent/blueprint.gained`
- `env_agent/blueprint.gained.activity_rate`
- `env_agent/blueprint.gained.activity_rate.std_dev`
- `env_agent/blueprint.gained.avg`
- `env_agent/blueprint.gained.avg.std_dev`
- `env_agent/blueprint.gained.first_step`
- `env_agent/blueprint.gained.first_step.std_dev`
- `env_agent/blueprint.gained.last_step`
- `env_agent/blueprint.gained.last_step.std_dev`
- `env_agent/blueprint.gained.max`
- `env_agent/blueprint.gained.max.std_dev`
- `env_agent/blueprint.gained.min`
- `env_agent/blueprint.gained.min.std_dev`
- `env_agent/blueprint.gained.rate`
- `env_agent/blueprint.gained.rate.std_dev`
- `env_agent/blueprint.gained.updates`
- `env_agent/blueprint.gained.updates.std_dev`
- `env_agent/blueprint.gained.std_dev`

**blueprint.get:** (9 values / 9 std_devs)
- `env_agent/blueprint.get`
- `env_agent/blueprint.get.activity_rate`
- `env_agent/blueprint.get.activity_rate.std_dev`
- `env_agent/blueprint.get.avg`
- `env_agent/blueprint.get.avg.std_dev`
- `env_agent/blueprint.get.first_step`
- `env_agent/blueprint.get.first_step.std_dev`
- `env_agent/blueprint.get.last_step`
- `env_agent/blueprint.get.last_step.std_dev`
- `env_agent/blueprint.get.max`
- `env_agent/blueprint.get.max.std_dev`
- `env_agent/blueprint.get.min`
- `env_agent/blueprint.get.min.std_dev`
- `env_agent/blueprint.get.rate`
- `env_agent/blueprint.get.rate.std_dev`
- `env_agent/blueprint.get.updates`
- `env_agent/blueprint.get.updates.std_dev`
- `env_agent/blueprint.get.std_dev`

**blueprint.lost:** (8 values / 8 std_devs)
- `env_agent/blueprint.lost`
- `env_agent/blueprint.lost.avg`
- `env_agent/blueprint.lost.avg.std_dev`
- `env_agent/blueprint.lost.first_step`
- `env_agent/blueprint.lost.first_step.std_dev`
- `env_agent/blueprint.lost.last_step`
- `env_agent/blueprint.lost.last_step.std_dev`
- `env_agent/blueprint.lost.max`
- `env_agent/blueprint.lost.max.std_dev`
- `env_agent/blueprint.lost.min`
- `env_agent/blueprint.lost.min.std_dev`
- `env_agent/blueprint.lost.rate`
- `env_agent/blueprint.lost.rate.std_dev`
- `env_agent/blueprint.lost.updates`
- `env_agent/blueprint.lost.updates.std_dev`
- `env_agent/blueprint.lost.std_dev`

**blueprint.put:** (8 values / 8 std_devs)
- `env_agent/blueprint.put`
- `env_agent/blueprint.put.avg`
- `env_agent/blueprint.put.avg.std_dev`
- `env_agent/blueprint.put.first_step`
- `env_agent/blueprint.put.first_step.std_dev`
- `env_agent/blueprint.put.last_step`
- `env_agent/blueprint.put.last_step.std_dev`
- `env_agent/blueprint.put.max`
- `env_agent/blueprint.put.max.std_dev`
- `env_agent/blueprint.put.min`
- `env_agent/blueprint.put.min.std_dev`
- `env_agent/blueprint.put.rate`
- `env_agent/blueprint.put.rate.std_dev`
- `env_agent/blueprint.put.updates`
- `env_agent/blueprint.put.updates.std_dev`
- `env_agent/blueprint.put.std_dev`

**blueprint.stolen.agent:** (8 values / 8 std_devs)
- `env_agent/blueprint.stolen.agent`
- `env_agent/blueprint.stolen.agent.avg`
- `env_agent/blueprint.stolen.agent.avg.std_dev`
- `env_agent/blueprint.stolen.agent.first_step`
- `env_agent/blueprint.stolen.agent.first_step.std_dev`
- `env_agent/blueprint.stolen.agent.last_step`
- `env_agent/blueprint.stolen.agent.last_step.std_dev`
- `env_agent/blueprint.stolen.agent.max`
- `env_agent/blueprint.stolen.agent.max.std_dev`
- `env_agent/blueprint.stolen.agent.min`
- `env_agent/blueprint.stolen.agent.min.std_dev`
- `env_agent/blueprint.stolen.agent.rate`
- `env_agent/blueprint.stolen.agent.rate.std_dev`
- `env_agent/blueprint.stolen.agent.updates`
- `env_agent/blueprint.stolen.agent.updates.std_dev`
- `env_agent/blueprint.stolen.agent.std_dev`

**blueprint.stolen_from.agent:** (8 values / 8 std_devs)
- `env_agent/blueprint.stolen_from.agent`
- `env_agent/blueprint.stolen_from.agent.avg`
- `env_agent/blueprint.stolen_from.agent.avg.std_dev`
- `env_agent/blueprint.stolen_from.agent.first_step`
- `env_agent/blueprint.stolen_from.agent.first_step.std_dev`
- `env_agent/blueprint.stolen_from.agent.last_step`
- `env_agent/blueprint.stolen_from.agent.last_step.std_dev`
- `env_agent/blueprint.stolen_from.agent.max`
- `env_agent/blueprint.stolen_from.agent.max.std_dev`
- `env_agent/blueprint.stolen_from.agent.min`
- `env_agent/blueprint.stolen_from.agent.min.std_dev`
- `env_agent/blueprint.stolen_from.agent.rate`
- `env_agent/blueprint.stolen_from.agent.rate.std_dev`
- `env_agent/blueprint.stolen_from.agent.updates`
- `env_agent/blueprint.stolen_from.agent.updates.std_dev`
- `env_agent/blueprint.stolen_from.agent.std_dev`

**heart.gained:** (9 values / 9 std_devs)
- `env_agent/heart.gained`
- `env_agent/heart.gained.activity_rate`
- `env_agent/heart.gained.activity_rate.std_dev`
- `env_agent/heart.gained.avg`
- `env_agent/heart.gained.avg.std_dev`
- `env_agent/heart.gained.first_step`
- `env_agent/heart.gained.first_step.std_dev`
- `env_agent/heart.gained.last_step`
- `env_agent/heart.gained.last_step.std_dev`
- `env_agent/heart.gained.max`
- `env_agent/heart.gained.max.std_dev`
- `env_agent/heart.gained.min`
- `env_agent/heart.gained.min.std_dev`
- `env_agent/heart.gained.rate`
- `env_agent/heart.gained.rate.std_dev`
- `env_agent/heart.gained.updates`
- `env_agent/heart.gained.updates.std_dev`
- `env_agent/heart.gained.std_dev`

**heart.get:** (9 values / 9 std_devs)
- `env_agent/heart.get`
- `env_agent/heart.get.activity_rate`
- `env_agent/heart.get.activity_rate.std_dev`
- `env_agent/heart.get.avg`
- `env_agent/heart.get.avg.std_dev`
- `env_agent/heart.get.first_step`
- `env_agent/heart.get.first_step.std_dev`
- `env_agent/heart.get.last_step`
- `env_agent/heart.get.last_step.std_dev`
- `env_agent/heart.get.max`
- `env_agent/heart.get.max.std_dev`
- `env_agent/heart.get.min`
- `env_agent/heart.get.min.std_dev`
- `env_agent/heart.get.rate`
- `env_agent/heart.get.rate.std_dev`
- `env_agent/heart.get.updates`
- `env_agent/heart.get.updates.std_dev`
- `env_agent/heart.get.std_dev`

**heart.lost:** (9 values / 9 std_devs)
- `env_agent/heart.lost`
- `env_agent/heart.lost.activity_rate`
- `env_agent/heart.lost.activity_rate.std_dev`
- `env_agent/heart.lost.avg`
- `env_agent/heart.lost.avg.std_dev`
- `env_agent/heart.lost.first_step`
- `env_agent/heart.lost.first_step.std_dev`
- `env_agent/heart.lost.last_step`
- `env_agent/heart.lost.last_step.std_dev`
- `env_agent/heart.lost.max`
- `env_agent/heart.lost.max.std_dev`
- `env_agent/heart.lost.min`
- `env_agent/heart.lost.min.std_dev`
- `env_agent/heart.lost.rate`
- `env_agent/heart.lost.rate.std_dev`
- `env_agent/heart.lost.updates`
- `env_agent/heart.lost.updates.std_dev`
- `env_agent/heart.lost.std_dev`

**heart.put:** (9 values / 9 std_devs)
- `env_agent/heart.put`
- `env_agent/heart.put.activity_rate`
- `env_agent/heart.put.activity_rate.std_dev`
- `env_agent/heart.put.avg`
- `env_agent/heart.put.avg.std_dev`
- `env_agent/heart.put.first_step`
- `env_agent/heart.put.first_step.std_dev`
- `env_agent/heart.put.last_step`
- `env_agent/heart.put.last_step.std_dev`
- `env_agent/heart.put.max`
- `env_agent/heart.put.max.std_dev`
- `env_agent/heart.put.min`
- `env_agent/heart.put.min.std_dev`
- `env_agent/heart.put.rate`
- `env_agent/heart.put.rate.std_dev`
- `env_agent/heart.put.updates`
- `env_agent/heart.put.updates.std_dev`
- `env_agent/heart.put.std_dev`

**heart.stolen.agent:** (8 values / 8 std_devs)
- `env_agent/heart.stolen.agent`
- `env_agent/heart.stolen.agent.avg`
- `env_agent/heart.stolen.agent.avg.std_dev`
- `env_agent/heart.stolen.agent.first_step`
- `env_agent/heart.stolen.agent.first_step.std_dev`
- `env_agent/heart.stolen.agent.last_step`
- `env_agent/heart.stolen.agent.last_step.std_dev`
- `env_agent/heart.stolen.agent.max`
- `env_agent/heart.stolen.agent.max.std_dev`
- `env_agent/heart.stolen.agent.min`
- `env_agent/heart.stolen.agent.min.std_dev`
- `env_agent/heart.stolen.agent.rate`
- `env_agent/heart.stolen.agent.rate.std_dev`
- `env_agent/heart.stolen.agent.updates`
- `env_agent/heart.stolen.agent.updates.std_dev`
- `env_agent/heart.stolen.agent.std_dev`

**heart.stolen_from.agent:** (8 values / 8 std_devs)
- `env_agent/heart.stolen_from.agent`
- `env_agent/heart.stolen_from.agent.avg`
- `env_agent/heart.stolen_from.agent.avg.std_dev`
- `env_agent/heart.stolen_from.agent.first_step`
- `env_agent/heart.stolen_from.agent.first_step.std_dev`
- `env_agent/heart.stolen_from.agent.last_step`
- `env_agent/heart.stolen_from.agent.last_step.std_dev`
- `env_agent/heart.stolen_from.agent.max`
- `env_agent/heart.stolen_from.agent.max.std_dev`
- `env_agent/heart.stolen_from.agent.min`
- `env_agent/heart.stolen_from.agent.min.std_dev`
- `env_agent/heart.stolen_from.agent.rate`
- `env_agent/heart.stolen_from.agent.rate.std_dev`
- `env_agent/heart.stolen_from.agent.updates`
- `env_agent/heart.stolen_from.agent.updates.std_dev`
- `env_agent/heart.stolen_from.agent.std_dev`

**laser.gained:** (9 values / 9 std_devs)
- `env_agent/laser.gained`
- `env_agent/laser.gained.activity_rate`
- `env_agent/laser.gained.activity_rate.std_dev`
- `env_agent/laser.gained.avg`
- `env_agent/laser.gained.avg.std_dev`
- `env_agent/laser.gained.first_step`
- `env_agent/laser.gained.first_step.std_dev`
- `env_agent/laser.gained.last_step`
- `env_agent/laser.gained.last_step.std_dev`
- `env_agent/laser.gained.max`
- `env_agent/laser.gained.max.std_dev`
- `env_agent/laser.gained.min`
- `env_agent/laser.gained.min.std_dev`
- `env_agent/laser.gained.rate`
- `env_agent/laser.gained.rate.std_dev`
- `env_agent/laser.gained.updates`
- `env_agent/laser.gained.updates.std_dev`
- `env_agent/laser.gained.std_dev`

**laser.get:** (9 values / 9 std_devs)
- `env_agent/laser.get`
- `env_agent/laser.get.activity_rate`
- `env_agent/laser.get.activity_rate.std_dev`
- `env_agent/laser.get.avg`
- `env_agent/laser.get.avg.std_dev`
- `env_agent/laser.get.first_step`
- `env_agent/laser.get.first_step.std_dev`
- `env_agent/laser.get.last_step`
- `env_agent/laser.get.last_step.std_dev`
- `env_agent/laser.get.max`
- `env_agent/laser.get.max.std_dev`
- `env_agent/laser.get.min`
- `env_agent/laser.get.min.std_dev`
- `env_agent/laser.get.rate`
- `env_agent/laser.get.rate.std_dev`
- `env_agent/laser.get.updates`
- `env_agent/laser.get.updates.std_dev`
- `env_agent/laser.get.std_dev`

**laser.lost:** (9 values / 9 std_devs)
- `env_agent/laser.lost`
- `env_agent/laser.lost.activity_rate`
- `env_agent/laser.lost.activity_rate.std_dev`
- `env_agent/laser.lost.avg`
- `env_agent/laser.lost.avg.std_dev`
- `env_agent/laser.lost.first_step`
- `env_agent/laser.lost.first_step.std_dev`
- `env_agent/laser.lost.last_step`
- `env_agent/laser.lost.last_step.std_dev`
- `env_agent/laser.lost.max`
- `env_agent/laser.lost.max.std_dev`
- `env_agent/laser.lost.min`
- `env_agent/laser.lost.min.std_dev`
- `env_agent/laser.lost.rate`
- `env_agent/laser.lost.rate.std_dev`
- `env_agent/laser.lost.updates`
- `env_agent/laser.lost.updates.std_dev`
- `env_agent/laser.lost.std_dev`

**laser.stolen.agent:** (8 values / 8 std_devs)
- `env_agent/laser.stolen.agent`
- `env_agent/laser.stolen.agent.avg`
- `env_agent/laser.stolen.agent.avg.std_dev`
- `env_agent/laser.stolen.agent.first_step`
- `env_agent/laser.stolen.agent.first_step.std_dev`
- `env_agent/laser.stolen.agent.last_step`
- `env_agent/laser.stolen.agent.last_step.std_dev`
- `env_agent/laser.stolen.agent.max`
- `env_agent/laser.stolen.agent.max.std_dev`
- `env_agent/laser.stolen.agent.min`
- `env_agent/laser.stolen.agent.min.std_dev`
- `env_agent/laser.stolen.agent.rate`
- `env_agent/laser.stolen.agent.rate.std_dev`
- `env_agent/laser.stolen.agent.updates`
- `env_agent/laser.stolen.agent.updates.std_dev`
- `env_agent/laser.stolen.agent.std_dev`

**laser.stolen_from.agent:** (8 values / 8 std_devs)
- `env_agent/laser.stolen_from.agent`
- `env_agent/laser.stolen_from.agent.avg`
- `env_agent/laser.stolen_from.agent.avg.std_dev`
- `env_agent/laser.stolen_from.agent.first_step`
- `env_agent/laser.stolen_from.agent.first_step.std_dev`
- `env_agent/laser.stolen_from.agent.last_step`
- `env_agent/laser.stolen_from.agent.last_step.std_dev`
- `env_agent/laser.stolen_from.agent.max`
- `env_agent/laser.stolen_from.agent.max.std_dev`
- `env_agent/laser.stolen_from.agent.min`
- `env_agent/laser.stolen_from.agent.min.std_dev`
- `env_agent/laser.stolen_from.agent.rate`
- `env_agent/laser.stolen_from.agent.rate.std_dev`
- `env_agent/laser.stolen_from.agent.updates`
- `env_agent/laser.stolen_from.agent.updates.std_dev`
- `env_agent/laser.stolen_from.agent.std_dev`

**ore.red.gained:** (9 values / 9 std_devs)
- `env_agent/ore.red.gained`
- `env_agent/ore.red.gained.activity_rate`
- `env_agent/ore.red.gained.activity_rate.std_dev`
- `env_agent/ore.red.gained.avg`
- `env_agent/ore.red.gained.avg.std_dev`
- `env_agent/ore.red.gained.first_step`
- `env_agent/ore.red.gained.first_step.std_dev`
- `env_agent/ore.red.gained.last_step`
- `env_agent/ore.red.gained.last_step.std_dev`
- `env_agent/ore.red.gained.max`
- `env_agent/ore.red.gained.max.std_dev`
- `env_agent/ore.red.gained.min`
- `env_agent/ore.red.gained.min.std_dev`
- `env_agent/ore.red.gained.rate`
- `env_agent/ore.red.gained.rate.std_dev`
- `env_agent/ore.red.gained.updates`
- `env_agent/ore.red.gained.updates.std_dev`
- `env_agent/ore.red.gained.std_dev`

**ore.red.get:** (9 values / 9 std_devs)
- `env_agent/ore.red.get`
- `env_agent/ore.red.get.activity_rate`
- `env_agent/ore.red.get.activity_rate.std_dev`
- `env_agent/ore.red.get.avg`
- `env_agent/ore.red.get.avg.std_dev`
- `env_agent/ore.red.get.first_step`
- `env_agent/ore.red.get.first_step.std_dev`
- `env_agent/ore.red.get.last_step`
- `env_agent/ore.red.get.last_step.std_dev`
- `env_agent/ore.red.get.max`
- `env_agent/ore.red.get.max.std_dev`
- `env_agent/ore.red.get.min`
- `env_agent/ore.red.get.min.std_dev`
- `env_agent/ore.red.get.rate`
- `env_agent/ore.red.get.rate.std_dev`
- `env_agent/ore.red.get.updates`
- `env_agent/ore.red.get.updates.std_dev`
- `env_agent/ore.red.get.std_dev`

**ore.red.lost:** (9 values / 9 std_devs)
- `env_agent/ore.red.lost`
- `env_agent/ore.red.lost.activity_rate`
- `env_agent/ore.red.lost.activity_rate.std_dev`
- `env_agent/ore.red.lost.avg`
- `env_agent/ore.red.lost.avg.std_dev`
- `env_agent/ore.red.lost.first_step`
- `env_agent/ore.red.lost.first_step.std_dev`
- `env_agent/ore.red.lost.last_step`
- `env_agent/ore.red.lost.last_step.std_dev`
- `env_agent/ore.red.lost.max`
- `env_agent/ore.red.lost.max.std_dev`
- `env_agent/ore.red.lost.min`
- `env_agent/ore.red.lost.min.std_dev`
- `env_agent/ore.red.lost.rate`
- `env_agent/ore.red.lost.rate.std_dev`
- `env_agent/ore.red.lost.updates`
- `env_agent/ore.red.lost.updates.std_dev`
- `env_agent/ore.red.lost.std_dev`

**ore.red.put:** (9 values / 9 std_devs)
- `env_agent/ore.red.put`
- `env_agent/ore.red.put.activity_rate`
- `env_agent/ore.red.put.activity_rate.std_dev`
- `env_agent/ore.red.put.avg`
- `env_agent/ore.red.put.avg.std_dev`
- `env_agent/ore.red.put.first_step`
- `env_agent/ore.red.put.first_step.std_dev`
- `env_agent/ore.red.put.last_step`
- `env_agent/ore.red.put.last_step.std_dev`
- `env_agent/ore.red.put.max`
- `env_agent/ore.red.put.max.std_dev`
- `env_agent/ore.red.put.min`
- `env_agent/ore.red.put.min.std_dev`
- `env_agent/ore.red.put.rate`
- `env_agent/ore.red.put.rate.std_dev`
- `env_agent/ore.red.put.updates`
- `env_agent/ore.red.put.updates.std_dev`
- `env_agent/ore.red.put.std_dev`

**ore.red.stolen.agent:** (8 values / 8 std_devs)
- `env_agent/ore.red.stolen.agent`
- `env_agent/ore.red.stolen.agent.avg`
- `env_agent/ore.red.stolen.agent.avg.std_dev`
- `env_agent/ore.red.stolen.agent.first_step`
- `env_agent/ore.red.stolen.agent.first_step.std_dev`
- `env_agent/ore.red.stolen.agent.last_step`
- `env_agent/ore.red.stolen.agent.last_step.std_dev`
- `env_agent/ore.red.stolen.agent.max`
- `env_agent/ore.red.stolen.agent.max.std_dev`
- `env_agent/ore.red.stolen.agent.min`
- `env_agent/ore.red.stolen.agent.min.std_dev`
- `env_agent/ore.red.stolen.agent.rate`
- `env_agent/ore.red.stolen.agent.rate.std_dev`
- `env_agent/ore.red.stolen.agent.updates`
- `env_agent/ore.red.stolen.agent.updates.std_dev`
- `env_agent/ore.red.stolen.agent.std_dev`

**ore.red.stolen_from.agent:** (8 values / 8 std_devs)
- `env_agent/ore.red.stolen_from.agent`
- `env_agent/ore.red.stolen_from.agent.avg`
- `env_agent/ore.red.stolen_from.agent.avg.std_dev`
- `env_agent/ore.red.stolen_from.agent.first_step`
- `env_agent/ore.red.stolen_from.agent.first_step.std_dev`
- `env_agent/ore.red.stolen_from.agent.last_step`
- `env_agent/ore.red.stolen_from.agent.last_step.std_dev`
- `env_agent/ore.red.stolen_from.agent.max`
- `env_agent/ore.red.stolen_from.agent.max.std_dev`
- `env_agent/ore.red.stolen_from.agent.min`
- `env_agent/ore.red.stolen_from.agent.min.std_dev`
- `env_agent/ore.red.stolen_from.agent.rate`
- `env_agent/ore.red.stolen_from.agent.rate.std_dev`
- `env_agent/ore.red.stolen_from.agent.updates`
- `env_agent/ore.red.stolen_from.agent.updates.std_dev`
- `env_agent/ore.red.stolen_from.agent.std_dev`

**status.frozen.ticks:** (9 values / 9 std_devs)
- `env_agent/status.frozen.ticks`
- `env_agent/status.frozen.ticks.activity_rate`
- `env_agent/status.frozen.ticks.activity_rate.std_dev`
- `env_agent/status.frozen.ticks.avg`
- `env_agent/status.frozen.ticks.avg.std_dev`
- `env_agent/status.frozen.ticks.first_step`
- `env_agent/status.frozen.ticks.first_step.std_dev`
- `env_agent/status.frozen.ticks.last_step`
- `env_agent/status.frozen.ticks.last_step.std_dev`
- `env_agent/status.frozen.ticks.max`
- `env_agent/status.frozen.ticks.max.std_dev`
- `env_agent/status.frozen.ticks.min`
- `env_agent/status.frozen.ticks.min.std_dev`
- `env_agent/status.frozen.ticks.rate`
- `env_agent/status.frozen.ticks.rate.std_dev`
- `env_agent/status.frozen.ticks.updates`
- `env_agent/status.frozen.ticks.updates.std_dev`
- `env_agent/status.frozen.ticks.std_dev`

**status.frozen.ticks.agent:** (9 values / 9 std_devs)
- `env_agent/status.frozen.ticks.agent`
- `env_agent/status.frozen.ticks.agent.activity_rate`
- `env_agent/status.frozen.ticks.agent.activity_rate.std_dev`
- `env_agent/status.frozen.ticks.agent.avg`
- `env_agent/status.frozen.ticks.agent.avg.std_dev`
- `env_agent/status.frozen.ticks.agent.first_step`
- `env_agent/status.frozen.ticks.agent.first_step.std_dev`
- `env_agent/status.frozen.ticks.agent.last_step`
- `env_agent/status.frozen.ticks.agent.last_step.std_dev`
- `env_agent/status.frozen.ticks.agent.max`
- `env_agent/status.frozen.ticks.agent.max.std_dev`
- `env_agent/status.frozen.ticks.agent.min`
- `env_agent/status.frozen.ticks.agent.min.std_dev`
- `env_agent/status.frozen.ticks.agent.rate`
- `env_agent/status.frozen.ticks.agent.rate.std_dev`
- `env_agent/status.frozen.ticks.agent.updates`
- `env_agent/status.frozen.ticks.agent.updates.std_dev`
- `env_agent/status.frozen.ticks.agent.std_dev`


