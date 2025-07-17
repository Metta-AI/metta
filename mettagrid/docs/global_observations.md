# Global Observations

Global observations are special observation tokens that provide environment-wide information to agents, as opposed to object-specific observations.

## Game Rewards Observation

The `game_rewards` global observation provides agents with information about the reward structure of the current game. This allows agents to adapt their behavior based on which resources are valuable.

### Enabling Game Rewards

To enable the game rewards observation, set `game.global_obs.game_rewards` to `true` in your configuration:

```yaml
game:
  global_obs:
    game_rewards: true
```

### How It Works

When enabled, a special observation token is added to each agent's observation with:
- **Feature ID**: 13 (`ObservationFeature::GameRewards`)
- **Location**: Center of the observation window
- **Value**: Packed representation of resource rewards

The observation packs rewards for four key resources into a single byte:
- **Ore** (any ore type): Bits 7-6
- **Battery** (any battery type): Bits 5-4
- **Laser**: Bits 3-2
- **Armor**: Bits 1-0

Each 2-bit value represents a quantized reward level:
- `0`: No reward (reward ≤ 0)
- `1`: Low reward (0 < reward ≤ 0.5)
- `2`: Medium reward (0.5 < reward ≤ 1.0)
- `3`: High reward (reward > 1.0)

### Example

With the following reward configuration:
```yaml
agent:
  rewards:
    ore_red: 1.0      # Quantized to 2
    battery_blue: 0.3 # Quantized to 1
    laser: 0.8        # Quantized to 2
    armor: 1.5        # Quantized to 3
```

The packed value would be:
- Binary: `10 01 10 11` = `0x9B` = 155
- Ore: 2 (medium)
- Battery: 1 (low)
- Laser: 2 (medium)
- Armor: 3 (high)

### Use Cases

This observation is useful for:
- **Multi-task learning**: Agents can learn policies that adapt to different reward structures
- **Curriculum learning**: Gradually changing rewards can guide agent behavior
- **Game variants**: Same environment with different objectives

### Implementation Details

- The packed reward value is computed once when the environment is created
- It remains constant throughout an episode
- If a resource type is not found in the inventory, its reward is treated as 0
- The first matching resource name is used (e.g., "ore_red" matches "ore")