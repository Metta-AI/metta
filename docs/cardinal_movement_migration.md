# Direct Movement Guide

## Overview

The direct movement system provides alternatives to traditional tank-style controls (forward/backward + rotate). We now support:
- **Cardinal movement**: 4-directional movement (North/South/East/West)
- **8-directional movement**: Movement in 8 directions including diagonals (N/NE/E/SE/S/SW/W/NW)

All movement types can coexist in the same environment, giving maximum flexibility.

## Key Changes

### Movement Actions

**Tank-style (relative) movement:**
- `move` action with arg 0: Move forward (relative to agent orientation)
- `move` action with arg 1: Move backward (relative to agent orientation)
- `rotate` action with args 0-3: Set orientation (Up/Down/Left/Right)

**Cardinal movement (4-directional):**
- `move_cardinal` action with arg 0: Move North (Up)
- `move_cardinal` action with arg 1: Move South (Down)
- `move_cardinal` action with arg 2: Move West (Left)
- `move_cardinal` action with arg 3: Move East (Right)

**8-directional movement:**
- `move_8directional` action with arg 0: Move North
- `move_8directional` action with arg 1: Move Northeast
- `move_8directional` action with arg 2: Move East
- `move_8directional` action with arg 3: Move Southeast
- `move_8directional` action with arg 4: Move South
- `move_8directional` action with arg 5: Move Southwest
- `move_8directional` action with arg 6: Move West
- `move_8directional` action with arg 7: Move Northwest

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
    
    # Cardinal movement (4-directional)
    move_cardinal:
      enabled: true  # or false
    
    # 8-directional movement
    move_8directional:
      enabled: true  # or false
```

### Agent Orientation

**Important change**: Direct movement actions (`move_cardinal` and `move_8directional`) now perform single atomic movements:
- They do NOT change the agent's orientation
- Orientation is only changed by the `rotate` action
- This ensures each action does exactly one thing
- Orientation still affects directional actions like `attack`

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

### Pure Cardinal Movement (4-directional)

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

### Pure 8-Directional Movement

```yaml
defaults:
  - mettagrid

game:
  actions:
    move:
      enabled: false
    rotate:
      enabled: false
    move_8directional:
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

Direct movement systems offer different trade-offs:

**Cardinal movement (4-directional)**:
- Smaller action space than tank-style (4 vs 2+4 for move+rotate)
- More direct pathfinding than tank-style
- Good for grid-aligned environments

**8-directional movement**:
- Larger action space (8 directions)
- Most efficient pathfinding (diagonal shortcuts)
- Best for open environments with fewer obstacles

**Tank-style movement**:
- Smallest individual action spaces (2 for move, 4 for rotate)
- Requires action sequences for diagonal movement
- Natural for environments with facing-dependent mechanics
