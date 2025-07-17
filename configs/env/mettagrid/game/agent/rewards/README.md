# Agent Reward Configuration

The agent reward system in MettaGrid supports two types of rewards:

## 1. Inventory Rewards
Located under `rewards.inventory`, these rewards are based on items in the agent's inventory:

```yaml
rewards:
  inventory:
    ore_red: 0.1      # 0.1 reward per ore_red in inventory
    ore_red_max: 10   # Maximum reward for ore_red
    heart: 1.0        # 1.0 reward per heart
    heart_max: null   # No limit on heart rewards
```

### How Inventory Rewards Work:
- Agents receive (or lose) when gaining or losing items
- When an item is gained: reward increases by `item_reward * quantity` (up to max)
- When an item is lost: reward decreases by the same amount
- The `_max` suffix sets a cap on how much reward is given for the item

## 2. Stats Rewards (Future Feature)
Located under `rewards.stats`, these rewards are based on agent statistics tracked by StatsTracker:

```yaml
rewards:
  stats:
    action.attack.success: 0.1         # 0.1 reward per successful attack
    action.attack.success_max: 10      # Max reward for attacks
    ore_red.gained: 0.01               # 0.01 reward each time ore is gained
    action.move.success: 0.001         # Small reward for movement
```

### Available Stats:
Stats are automatically tracked for all agents and include:
- **Actions**: `action.<action_name>.success`, `action.<action_name>.failed`
- **Inventory Changes**: `<item_name>.gained`, `<item_name>.lost`, `<item_name>.put`, `<item_name>.get`
- **Combat**: `action.attack.<group>.hit.<target_group>`, `action.attack.<group>.blocked_by.<target_group>`
- **Status**: `status.frozen.ticks`, `action.failure_penalty`
- **Custom Stats**: Any stat tracked via `agent.stats.add()` or `agent.stats.incr()`

## Example Configurations

### Hearts Only (Default)
```yaml
inventory:
  heart: 1
  heart_max: null
```
