# Cardinal Movement Guide

## Overview

The cardinal movement system provides direct movement in cardinal directions (North/South/East/West) as an alternative to traditional tank-style controls (forward/backward + rotate). Both movement types can coexist in the same environment, giving maximum flexibility.

## Key Changes

### Movement Actions

**Tank-style (relative) movement:**
- `move` action with arg 0: Move forward (relative to agent orientation)
- `move` action with arg 1: Move backward (relative to agent orientation)
- `rotate` action with args 0-3: Set orientation (Up/Down/Left/Right)

**Cardinal movement:**
- `move_cardinal` action with arg 0: Move North (Up)
- `move_cardinal` action with arg 1: Move South (Down)
- `move_cardinal` action with arg 2: Move West (Left)
- `move_cardinal` action with arg 3: Move East (Right)

### Configuration

Movement types are controlled through the actions configuration:

```yaml
game:
  actions:
    # Tank-style movement
    move:
      enabled: true  # or false
    rotate:
      enabled: true  # or false
    
    # Cardinal movement
    move_cardinal:
      enabled: true  # or false
```

### Agent Orientation

- In cardinal mode, agents still maintain an orientation for actions like `attack` that depend on facing direction
- **Agents always rotate to face the direction they're trying to move, even if the movement is blocked**
- This allows agents to "look at" walls and obstacles
- Initial orientation remains Up (0) by default

## Migration Path for Trained Policies

### Configuration-Based Migration

The simplest migration path is to enable the appropriate actions in your configuration:

1. **For existing tank-style policies**: Keep `move` and `rotate` enabled, disable `move_cardinal`
2. **For new cardinal-style policies**: Disable `move` and `rotate`, enable `move_cardinal`
3. **For experiments**: Enable all three to compare behaviors

### Action Space Considerations

When changing movement types, be aware that the action indices will change:

```python
# Example with tank-style only
action_names = ["noop", "move", "rotate", "attack"]  
# move is index 1, rotate is index 2

# Example with cardinal only
action_names = ["noop", "move_cardinal", "attack"]
# move_cardinal is index 1, attack is index 2

# Example with both enabled
action_names = ["noop", "move", "rotate", "move_cardinal", "attack"]
# move is index 1, rotate is index 2, move_cardinal is index 3
```

## Retraining Considerations

### Action Space Changes

When training policies with cardinal movement:
1. The `move_cardinal` action has 4 arguments (0-3) for the four directions
2. The action space size depends on which actions are enabled
3. Policies can be trained with just one movement type or both

### Behavioral Differences

- **Efficiency**: Cardinal movement can be more efficient as it eliminates the need for rotation actions
- **Exploration**: Random action policies will behave differently - they'll move in all directions equally rather than mostly forward/backward
- **Strategies**: Policies may need to learn new strategies that don't rely on maintaining specific orientations

### Training Tips

1. **Curriculum Learning**: Start with simple navigation tasks before complex scenarios
2. **Reward Shaping**: Consider rewards that encourage efficient pathfinding
3. **Action Masking**: Mask invalid directions (e.g., into walls) to speed up learning

## Example Configurations

### Pure Cardinal Movement

```yaml
defaults:
  - mettagrid

game:
  actions:
    move:
      enabled: false
    rotate:
      enabled: false
    move_cardinal:
      enabled: true
```

### Pure Tank-Style Movement (Default)

```yaml
defaults:
  - mettagrid

game:
  actions:
    move:
      enabled: true
    rotate:
      enabled: true
    move_cardinal:
      enabled: false
```

### Hybrid Movement Mode

```yaml
defaults:
  - mettagrid

game:
  actions:
    move:
      enabled: true
    rotate:
      enabled: true
    move_cardinal:
      enabled: true
```

## Backward Compatibility

The system maintains full backward compatibility:
- Existing configurations work unchanged (tank-style movement is the default)
- New `move_cardinal` action is disabled by default
- Policies trained before this change continue to work exactly as before

## Performance Considerations

Cardinal movement may offer performance benefits:
- Reduced action space (no rotate action)
- More direct pathfinding
- Simpler action selection logic

However, some scenarios designed for tank-style movement may need adjustment to work well with cardinal movement.
