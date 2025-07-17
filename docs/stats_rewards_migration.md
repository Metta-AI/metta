# Stats-Based Rewards Migration Guide

## Overview

The reward configuration has been restructured to support two types of rewards:
1. **Inventory rewards**: Rewards based on items in the agent's inventory (existing functionality)
2. **Stats rewards**: Rewards based on tracked agent statistics (new functionality)

## Configuration Changes

### Old Format
```yaml
game:
  agent:
    rewards:
      ore_red: 0.1
      ore_red_max: 1
      heart: 1.0
```

### New Format
```yaml
game:
  agent:
    rewards:
      inventory:
        ore_red: 0.1
        ore_red_max: 1
        heart: 1.0
      stats:
        action.move.success: 0.01      # 0.01 reward per successful move
        action.move.success_max: 1.0   # Max 1.0 total reward from moves
        action.attack.success: 0.5     # 0.5 reward per successful attack
        action.failure_penalty: -0.1   # -0.1 penalty per failed action
```

## Backward Compatibility

The system automatically migrates old configurations to the new format. If you have direct reward keys under `rewards:`, they will be moved to `rewards.inventory:`.

## Available Stats

Stats are tracked by the `StatsTracker` and include:

### Action Stats
- `action.{action_name}.success` - Number of successful actions
- `action.{action_name}.failed` - Number of failed actions
- `action.failure_penalty` - Number of action failures

### Inventory Stats
- `{item_name}.gained` - Amount of item gained
- `{item_name}.lost` - Amount of item lost
- `{item_name}.get` - Items retrieved from objects
- `{item_name}.put` - Items placed into objects

### Combat Stats
- `action.attack.{group1}.hit.{group2}` - Attacks by group1 that hit group2
- `action.attack.{group}.friendly_fire` - Friendly fire incidents
- `action.attack.{group}.blocked_by.{other_group}` - Blocked attacks

### Other Stats
- `status.frozen.ticks` - Time spent frozen
- Any custom stats added by the game

## Example Configuration

Here's a complete example showing both inventory and stats rewards:

```yaml
game:
  agent:
    rewards:
      inventory:
        # Standard inventory rewards
        heart: 1.0
        ore_red: 0.01
        ore_red_max: 10
        battery_red: 0.02
        battery_red_max: 10
      
      stats:
        # Movement rewards
        action.move.success: 0.001
        action.move.success_max: 1.0
        
        # Combat rewards
        action.attack.success: 0.1
        action.attack.success_max: 5.0
        
        # Collection bonuses (in addition to inventory rewards)
        ore_red.gained: 0.005
        battery_red.gained: 0.01
        
        # Penalties
        action.failure_penalty: -0.01
        status.frozen.ticks: -0.001
```

## Implementation Details

1. Stats rewards are computed at the end of each step, after all actions have been processed
2. Both `inventory` and `stats` rewards are additive - agents receive both types
3. The `_max` suffix works the same for both types - it caps the total cumulative reward
4. Negative rewards (penalties) are supported for stats
5. Stats rewards use the exact stat names as tracked by `StatsTracker`

## C++ API Changes

The `AgentConfig` struct now includes:
```cpp
std::map<std::string, RewardType> stat_rewards;
std::map<std::string, float> stat_reward_max;
```

The `Agent` class has a new method:
```cpp
void compute_stat_rewards();
```

This is called automatically at the end of each step.