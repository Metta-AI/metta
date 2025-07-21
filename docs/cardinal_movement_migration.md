# Cardinal Movement Migration Guide

## Overview

The cardinal movement system replaces the traditional tank-style controls (forward/backward + rotate) with direct movement in cardinal directions (North/South/East/West). This change simplifies agent control and makes movement more intuitive.

## Key Changes

### Movement Actions

**Tank-style (relative) movement:**
- `move` action with arg 0: Move forward (relative to agent orientation)
- `move` action with arg 1: Move backward (relative to agent orientation)
- `rotate` action with args 0-3: Set orientation (Up/Down/Left/Right)

**Cardinal movement:**
- `move` action with arg 0: Move North (Up)
- `move` action with arg 1: Move South (Down)
- `move` action with arg 2: Move West (Left)
- `move` action with arg 3: Move East (Right)
- `rotate` action: Not available in cardinal mode

### Configuration

To enable cardinal movement, add the following to your environment configuration:

```yaml
game:
  movement_mode: cardinal  # Options: "relative" (default) or "cardinal"
```

### Agent Orientation

- In cardinal mode, agents still maintain an orientation for actions like `attack` that depend on facing direction
- The orientation no longer affects movement direction
- Initial orientation remains Up (0) by default

## Migration Path for Trained Policies

### Using the Action Remapper

For policies trained with tank-style controls, use the `TankToCardinalRemapper`:

```python
from metta.mettagrid.util.action_remapper import TankToCardinalRemapper

# Initialize remapper with your environment's action names
action_names = env.action_names()
remapper = TankToCardinalRemapper(action_names)

# Track agent orientations (get from env.grid_objects())
agent_orientations = {0: 0, 1: 2}  # agent_id -> orientation

# Remap actions before stepping
actions = policy.get_actions(observations)  # Your existing policy
remapped_actions = remapper.remap_actions(actions, agent_orientations)
env.step(remapped_actions)
```

### Conversion Examples

| Tank-style Action | Agent Orientation | Cardinal Equivalent |
|-------------------|-------------------|-------------------|
| Move forward | Facing Up | Move North |
| Move forward | Facing Right | Move East |
| Move backward | Facing Up | Move South |
| Rotate to Left | Any | Move West |

## Retraining Considerations

### Action Space Changes

When retraining policies for cardinal movement:
1. The `move` action now has 4 arguments (0-3) instead of 2 (0-1)
2. The `rotate` action is no longer available
3. Total action space may be smaller (one less action type)

### Behavioral Differences

- **Efficiency**: Cardinal movement can be more efficient as it eliminates the need for rotation actions
- **Exploration**: Random action policies will behave differently - they'll move in all directions equally rather than mostly forward/backward
- **Strategies**: Policies may need to learn new strategies that don't rely on maintaining specific orientations

### Training Tips

1. **Curriculum Learning**: Start with simple navigation tasks before complex scenarios
2. **Reward Shaping**: Consider rewards that encourage efficient pathfinding
3. **Action Masking**: Mask invalid directions (e.g., into walls) to speed up learning

## Example Configurations

### Basic Cardinal Movement Environment

```yaml
defaults:
  - game/agent: agent
  - game/groups: solo
  - game/objects:
      - basic

game:
  num_agents: 1
  movement_mode: cardinal
  
  actions:
    noop:
      enabled: true
    move:
      enabled: true
    # rotate not needed in cardinal mode
    attack:
      enabled: true
```

### Multi-Agent Cardinal Environment

```yaml
game:
  num_agents: 4
  movement_mode: cardinal
  
  groups:
    team_a:
      id: 0
      group_reward_pct: 0.5
    team_b:
      id: 1
      group_reward_pct: 0.5
```

## Backward Compatibility

The system maintains full backward compatibility:
- Existing configurations default to `movement_mode: relative`
- Tank-style policies can still run unchanged in relative mode
- The action remapper provides a bridge for gradual migration

## Performance Considerations

Cardinal movement may offer performance benefits:
- Reduced action space (no rotate action)
- More direct pathfinding
- Simpler action selection logic

However, some scenarios designed for tank-style movement may need adjustment to work well with cardinal movement.