# Cyclical Converter Implementation

## Summary

Implemented a simple cyclical converter feature that automatically empties its inventory after a cooldown period. This creates timing-based gameplay elements where resources are only available during specific windows.

## What Was Changed

1. **Modified `converter.hpp`** (4 lines added):
   - Added auto-emptying logic in `finish_cooldown()` for type IDs 100-199
   - Tracks auto-empty events with `stats.incr("inventory.auto_emptied")`

2. **Created cyclical converter config** (`configs/env/mettagrid/game/objects/cyclical.yaml`):
   - Defined cyclical converter types with IDs 100-110
   - Various timing configurations for different gameplay scenarios

3. **Created test environment** (`configs/env/mettagrid/cyclical_test.yaml`):
   - Simple map with both regular and cyclical converters
   - Allows direct comparison of behaviors

4. **Created test map** (`configs/env/mettagrid/maps/debug/cyclical_test.map`):
   - 9x9 grid with agent and converters
   - Visual layout for easy testing

## How It Works

1. Cyclical converters (type_id 100-199) behave like normal converters
2. After cooldown completes, they automatically empty their inventory
3. Then immediately start producing again (if inputs available)

## Testing

Run this command to see it in action:
```bash
./tools/play.py env=env/mettagrid/cyclical_test cmd=play
```

Features:
- Regular converters (c) keep their hearts indefinitely
- Cyclical converters (C) auto-empty after cooldown
- Creates timing windows for resource collection

## Use Cases

1. **Timing Puzzles**: Resources only available at specific times
2. **Forced Movement**: Agents must patrol between converters
3. **Coordination**: Teams must synchronize collection timing
4. **Resource Scarcity**: Prevents hoarding by auto-removal

## Configuration Example

```yaml
cyclical_converter:
  type_id: 100  # Must be 100-199 for cyclical behavior
  input_resources: {}
  output_resources:
    heart: 1
  conversion_ticks: 10  # Time to produce
  cooldown: 20         # Time before auto-empty
  max_output: 1
  initial_resource_count: 0
  color: 13
```

Total implementation: ~10 lines of C++ code + configuration files
