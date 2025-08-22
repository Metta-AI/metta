# Env Game Metrics

## Overview

Game object counts and token tracking

**Total metrics in this section:** 214

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

**Count:** 214 metrics

**objects.altar:** (8 values / 8 std_devs)
- `env_game/objects.altar`
- `env_game/objects.altar.avg`
- `env_game/objects.altar.avg.std_dev`
- `env_game/objects.altar.first_step`
- `env_game/objects.altar.first_step.std_dev`
- `env_game/objects.altar.last_step`
- `env_game/objects.altar.last_step.std_dev`
- `env_game/objects.altar.max`
- `env_game/objects.altar.max.std_dev`
- `env_game/objects.altar.min`
- `env_game/objects.altar.min.std_dev`
- `env_game/objects.altar.rate`
- `env_game/objects.altar.rate.std_dev`
- `env_game/objects.altar.updates`
- `env_game/objects.altar.updates.std_dev`
- `env_game/objects.altar.std_dev`

**objects.armory:** (8 values / 8 std_devs)
- `env_game/objects.armory`
- `env_game/objects.armory.avg`
- `env_game/objects.armory.avg.std_dev`
- `env_game/objects.armory.first_step`
- `env_game/objects.armory.first_step.std_dev`
- `env_game/objects.armory.last_step`
- `env_game/objects.armory.last_step.std_dev`
- `env_game/objects.armory.max`
- `env_game/objects.armory.max.std_dev`
- `env_game/objects.armory.min`
- `env_game/objects.armory.min.std_dev`
- `env_game/objects.armory.rate`
- `env_game/objects.armory.rate.std_dev`
- `env_game/objects.armory.updates`
- `env_game/objects.armory.updates.std_dev`
- `env_game/objects.armory.std_dev`

**objects.block:** (8 values / 8 std_devs)
- `env_game/objects.block`
- `env_game/objects.block.avg`
- `env_game/objects.block.avg.std_dev`
- `env_game/objects.block.first_step`
- `env_game/objects.block.first_step.std_dev`
- `env_game/objects.block.last_step`
- `env_game/objects.block.last_step.std_dev`
- `env_game/objects.block.max`
- `env_game/objects.block.max.std_dev`
- `env_game/objects.block.min`
- `env_game/objects.block.min.std_dev`
- `env_game/objects.block.rate`
- `env_game/objects.block.rate.std_dev`
- `env_game/objects.block.updates`
- `env_game/objects.block.updates.std_dev`
- `env_game/objects.block.std_dev`

**objects.factory:** (8 values / 8 std_devs)
- `env_game/objects.factory`
- `env_game/objects.factory.avg`
- `env_game/objects.factory.avg.std_dev`
- `env_game/objects.factory.first_step`
- `env_game/objects.factory.first_step.std_dev`
- `env_game/objects.factory.last_step`
- `env_game/objects.factory.last_step.std_dev`
- `env_game/objects.factory.max`
- `env_game/objects.factory.max.std_dev`
- `env_game/objects.factory.min`
- `env_game/objects.factory.min.std_dev`
- `env_game/objects.factory.rate`
- `env_game/objects.factory.rate.std_dev`
- `env_game/objects.factory.updates`
- `env_game/objects.factory.updates.std_dev`
- `env_game/objects.factory.std_dev`

**objects.generator_red:** (8 values / 8 std_devs)
- `env_game/objects.generator_red`
- `env_game/objects.generator_red.avg`
- `env_game/objects.generator_red.avg.std_dev`
- `env_game/objects.generator_red.first_step`
- `env_game/objects.generator_red.first_step.std_dev`
- `env_game/objects.generator_red.last_step`
- `env_game/objects.generator_red.last_step.std_dev`
- `env_game/objects.generator_red.max`
- `env_game/objects.generator_red.max.std_dev`
- `env_game/objects.generator_red.min`
- `env_game/objects.generator_red.min.std_dev`
- `env_game/objects.generator_red.rate`
- `env_game/objects.generator_red.rate.std_dev`
- `env_game/objects.generator_red.updates`
- `env_game/objects.generator_red.updates.std_dev`
- `env_game/objects.generator_red.std_dev`

**objects.lab:** (8 values / 8 std_devs)
- `env_game/objects.lab`
- `env_game/objects.lab.avg`
- `env_game/objects.lab.avg.std_dev`
- `env_game/objects.lab.first_step`
- `env_game/objects.lab.first_step.std_dev`
- `env_game/objects.lab.last_step`
- `env_game/objects.lab.last_step.std_dev`
- `env_game/objects.lab.max`
- `env_game/objects.lab.max.std_dev`
- `env_game/objects.lab.min`
- `env_game/objects.lab.min.std_dev`
- `env_game/objects.lab.rate`
- `env_game/objects.lab.rate.std_dev`
- `env_game/objects.lab.updates`
- `env_game/objects.lab.updates.std_dev`
- `env_game/objects.lab.std_dev`

**objects.lasery:** (8 values / 8 std_devs)
- `env_game/objects.lasery`
- `env_game/objects.lasery.avg`
- `env_game/objects.lasery.avg.std_dev`
- `env_game/objects.lasery.first_step`
- `env_game/objects.lasery.first_step.std_dev`
- `env_game/objects.lasery.last_step`
- `env_game/objects.lasery.last_step.std_dev`
- `env_game/objects.lasery.max`
- `env_game/objects.lasery.max.std_dev`
- `env_game/objects.lasery.min`
- `env_game/objects.lasery.min.std_dev`
- `env_game/objects.lasery.rate`
- `env_game/objects.lasery.rate.std_dev`
- `env_game/objects.lasery.updates`
- `env_game/objects.lasery.updates.std_dev`
- `env_game/objects.lasery.std_dev`

**objects.mine_red:** (8 values / 8 std_devs)
- `env_game/objects.mine_red`
- `env_game/objects.mine_red.avg`
- `env_game/objects.mine_red.avg.std_dev`
- `env_game/objects.mine_red.first_step`
- `env_game/objects.mine_red.first_step.std_dev`
- `env_game/objects.mine_red.last_step`
- `env_game/objects.mine_red.last_step.std_dev`
- `env_game/objects.mine_red.max`
- `env_game/objects.mine_red.max.std_dev`
- `env_game/objects.mine_red.min`
- `env_game/objects.mine_red.min.std_dev`
- `env_game/objects.mine_red.rate`
- `env_game/objects.mine_red.rate.std_dev`
- `env_game/objects.mine_red.updates`
- `env_game/objects.mine_red.updates.std_dev`
- `env_game/objects.mine_red.std_dev`

**objects.temple:** (8 values / 8 std_devs)
- `env_game/objects.temple`
- `env_game/objects.temple.avg`
- `env_game/objects.temple.avg.std_dev`
- `env_game/objects.temple.first_step`
- `env_game/objects.temple.first_step.std_dev`
- `env_game/objects.temple.last_step`
- `env_game/objects.temple.last_step.std_dev`
- `env_game/objects.temple.max`
- `env_game/objects.temple.max.std_dev`
- `env_game/objects.temple.min`
- `env_game/objects.temple.min.std_dev`
- `env_game/objects.temple.rate`
- `env_game/objects.temple.rate.std_dev`
- `env_game/objects.temple.updates`
- `env_game/objects.temple.updates.std_dev`
- `env_game/objects.temple.std_dev`

**objects.wall:** (8 values / 8 std_devs)
- `env_game/objects.wall`
- `env_game/objects.wall.avg`
- `env_game/objects.wall.avg.std_dev`
- `env_game/objects.wall.first_step`
- `env_game/objects.wall.first_step.std_dev`
- `env_game/objects.wall.last_step`
- `env_game/objects.wall.last_step.std_dev`
- `env_game/objects.wall.max`
- `env_game/objects.wall.max.std_dev`
- `env_game/objects.wall.min`
- `env_game/objects.wall.min.std_dev`
- `env_game/objects.wall.rate`
- `env_game/objects.wall.rate.std_dev`
- `env_game/objects.wall.updates`
- `env_game/objects.wall.updates.std_dev`
- `env_game/objects.wall.std_dev`

**tokens_dropped:** (9 values / 9 std_devs)
- `env_game/tokens_dropped`
- `env_game/tokens_dropped.activity_rate`
- `env_game/tokens_dropped.activity_rate.std_dev`
- `env_game/tokens_dropped.avg`
- `env_game/tokens_dropped.avg.std_dev`
- `env_game/tokens_dropped.first_step`
- `env_game/tokens_dropped.first_step.std_dev`
- `env_game/tokens_dropped.last_step`
- `env_game/tokens_dropped.last_step.std_dev`
- `env_game/tokens_dropped.max`
- `env_game/tokens_dropped.max.std_dev`
- `env_game/tokens_dropped.min`
- `env_game/tokens_dropped.min.std_dev`
- `env_game/tokens_dropped.rate`
- `env_game/tokens_dropped.rate.std_dev`
- `env_game/tokens_dropped.updates`
- `env_game/tokens_dropped.updates.std_dev`
- `env_game/tokens_dropped.std_dev`

**tokens_free_space:** (9 values / 9 std_devs)
- `env_game/tokens_free_space`
- `env_game/tokens_free_space.activity_rate`
- `env_game/tokens_free_space.activity_rate.std_dev`
- `env_game/tokens_free_space.avg`
- `env_game/tokens_free_space.avg.std_dev`
- `env_game/tokens_free_space.first_step`
- `env_game/tokens_free_space.first_step.std_dev`
- `env_game/tokens_free_space.last_step`
- `env_game/tokens_free_space.last_step.std_dev`
- `env_game/tokens_free_space.max`
- `env_game/tokens_free_space.max.std_dev`
- `env_game/tokens_free_space.min`
- `env_game/tokens_free_space.min.std_dev`
- `env_game/tokens_free_space.rate`
- `env_game/tokens_free_space.rate.std_dev`
- `env_game/tokens_free_space.updates`
- `env_game/tokens_free_space.updates.std_dev`
- `env_game/tokens_free_space.std_dev`

**tokens_written:** (9 values / 9 std_devs)
- `env_game/tokens_written`
- `env_game/tokens_written.activity_rate`
- `env_game/tokens_written.activity_rate.std_dev`
- `env_game/tokens_written.avg`
- `env_game/tokens_written.avg.std_dev`
- `env_game/tokens_written.first_step`
- `env_game/tokens_written.first_step.std_dev`
- `env_game/tokens_written.last_step`
- `env_game/tokens_written.last_step.std_dev`
- `env_game/tokens_written.max`
- `env_game/tokens_written.max.std_dev`
- `env_game/tokens_written.min`
- `env_game/tokens_written.min.std_dev`
- `env_game/tokens_written.rate`
- `env_game/tokens_written.rate.std_dev`
- `env_game/tokens_written.updates`
- `env_game/tokens_written.updates.std_dev`
- `env_game/tokens_written.std_dev`


