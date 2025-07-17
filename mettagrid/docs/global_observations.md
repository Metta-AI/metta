# Global Observations

Global observations are special observation tokens that provide environment-wide information to agents, as opposed to object-specific observations.

## Inventory Rewards Observation

The inventory rewards global observation provides agents with information about which inventory items provide rewards. This allows agents to adapt their behavior based on which resources are valuable.

### How Inventory Rewards Work

The inventory rewards are automatically included as a global observation for each agent. Each agent sees their own reward configuration.

### How It Works

A special global observation token is added to each agent's observation with:
- **Location**: Center of the observation window
- **Feature ID**: 13 (`ObservationFeature::InventoryRewards`)
- **Value**: Packed representation of inventory rewards

The observation packs rewards for up to 8 inventory items into a single byte:
- **Item 0**: Bit 7 (MSB)
- **Item 1**: Bit 6
- **Item 2**: Bit 5
- **Item 3**: Bit 4
- **Item 4**: Bit 3
- **Item 5**: Bit 2
- **Item 6**: Bit 1
- **Item 7**: Bit 0 (LSB)

Each bit represents whether that inventory item has a positive reward:
- `0`: No reward (reward ≤ 0)
- `1`: Has reward (reward > 0)

### Example

With the following inventory and reward configuration:
```yaml
inventory_item_names: ["ore_red", "ore_blue", "battery", "laser", "armor", "heart", "gem", "crystal"]
agent:
  rewards:
    ore_red: 1.0      # Item 0: Has reward (bit = 1)
    ore_blue: 0.0     # Item 1: No reward (bit = 0)
    battery: 0.3      # Item 2: Has reward (bit = 1)
    laser: -0.5       # Item 3: No reward (bit = 0, negative)
    armor: 1.5        # Item 4: Has reward (bit = 1)
    heart: 0.0        # Item 5: No reward (bit = 0)
    gem: 0.1          # Item 6: Has reward (bit = 1)
    crystal: 2.0      # Item 7: Has reward (bit = 1)
```

The packed value would be:
- Binary: `10101011` = `0xAB` = 171
- Bit 7 (ore_red): 1
- Bit 6 (ore_blue): 0
- Bit 5 (battery): 1
- Bit 4 (laser): 0
- Bit 3 (armor): 1
- Bit 2 (heart): 0
- Bit 1 (gem): 1
- Bit 0 (crystal): 1

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
