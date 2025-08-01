# Movement Systems in Metta

This document describes the three movement systems available in Metta: tank-style, cardinal, and 8-way movement.

## Overview

Metta supports multiple movement systems that can be configured through command-line overrides:

1. **Tank Movement** (Default): Classic forward/backward + rotate controls
2. **Cardinal Movement**: Direct 4-directional movement (N/S/E/W)
3. **8-Way Movement**: Direct 8-directional movement including diagonals

All movement types can coexist in the same environment, giving maximum flexibility for different game mechanics and AI behaviors.

## Movement Types

### Tank Movement (Default)
- **Actions**: 
  - `move` (arg 0: forward, arg 1: backward) - moves relative to agent's facing direction
  - `rotate` (args 0-3: Up/Down/Left/Right) - changes agent orientation
- **Behavior**: Agents move in the direction they're facing, must rotate to change direction
- **Use Case**: Classic tank-style controls, good for combat scenarios where facing matters

### Cardinal Movement (4-directional)
- **Actions**: `move_cardinal` with arguments:
  - arg 0: Move North (Up)
  - arg 1: Move South (Down)
  - arg 2: Move West (Left)
  - arg 3: Move East (Right)
- **Behavior**: Direct movement in 4 cardinal directions, no rotation needed
- **Use Case**: Simplified movement for navigation tasks, grid-aligned environments

### 8-Way Movement
- **Actions**: `move_8way` with arguments:
  - arg 0: North
  - arg 1: Northeast
  - arg 2: East
  - arg 3: Southeast
  - arg 4: South
  - arg 5: Southwest
  - arg 6: West
  - arg 7: Northwest
- **Behavior**: Direct movement in 8 directions including diagonals, no rotation needed
- **Use Case**: Most flexible movement for complex navigation, open environments

## Key Concepts

### Agent Orientation

**Important**: Direct movement actions (`move_cardinal` and `move_8way`) perform single atomic movements:
- They do NOT change the agent's orientation
- Only the `rotate` action changes orientation
- This ensures each action does exactly one thing
- Orientation still affects directional actions like `attack`

### Action Space Changes

The enabled movement types affect the action space:

```python
# Tank-style only
action_names = ["noop", "move", "rotate", "attack"]  
# move is index 1, rotate is index 2

# Cardinal only
action_names = ["noop", "move_cardinal", "attack"]
# move_cardinal is index 1, attack is index 2

# Both enabled (hybrid mode)
action_names = ["noop", "move", "rotate", "move_cardinal", "attack"]
# move is index 1, rotate is index 2, move_cardinal is index 3
```

## Training Commands

Use command-line overrides to configure movement types during training:

### Tank Movement Training (Default)
```bash
uv run ./tools/train.py run=tank_movement_test \
  trainer.curriculum=/env/mettagrid/navigation/training/small \
  trainer.total_timesteps=500000 \
  trainer.num_workers=2
```

### Cardinal Movement Training
```bash
uv run ./tools/train.py run=cardinal_movement_test \
  trainer.curriculum=/env/mettagrid/navigation/training/small \
  ++trainer.env_overrides.game.actions.move.enabled=false \
  ++trainer.env_overrides.game.actions.rotate.enabled=false \
  ++trainer.env_overrides.game.actions.move_cardinal.enabled=true \
  trainer.total_timesteps=500000 \
  trainer.num_workers=2
```

### 8-Way Movement Training
```bash
uv run ./tools/train.py run=8way_movement_test \
  trainer.curriculum=/env/mettagrid/navigation/training/small \
  ++trainer.env_overrides.game.actions.move.enabled=false \
  ++trainer.env_overrides.game.actions.rotate.enabled=false \
  ++trainer.env_overrides.game.actions.move_8way.enabled=true \
  trainer.total_timesteps=500000 \
  trainer.num_workers=2
```

## Play/Evaluation Commands

Use similar overrides during evaluation to match the training configuration:

### Tank Movement Play
```bash
uv run ./tools/play.py run=play_tank \
  policy_uri=file://./train_dir/tank_movement_test/checkpoints \
  replay_job.sim.env=/env/mettagrid/arena/basic \
  device=cpu
```

### Cardinal Movement Play
```bash
uv run ./tools/play.py run=play_cardinal \
  policy_uri=file://./train_dir/cardinal_movement_test/checkpoints \
  replay_job.sim.env=/env/mettagrid/arena/basic \
  +replay_job.sim.env_overrides.game.actions.move.enabled=false \
  +replay_job.sim.env_overrides.game.actions.rotate.enabled=false \
  +replay_job.sim.env_overrides.game.actions.move_cardinal.enabled=true \
  device=cpu
```

### 8-Way Movement Play
```bash
uv run ./tools/play.py run=play_8way \
  policy_uri=file://./train_dir/8way_movement_test/checkpoints \
  replay_job.sim.env=/env/mettagrid/arena/basic \
  +replay_job.sim.env_overrides.game.actions.move.enabled=false \
  +replay_job.sim.env_overrides.game.actions.rotate.enabled=false \
  +replay_job.sim.env_overrides.game.actions.move_8way.enabled=true \
  device=cpu
```

## Performance Considerations

Each movement system offers different trade-offs:

### Cardinal Movement (4-directional)
- Smaller action space than tank-style (4 vs 2+4 for move+rotate)
- More direct pathfinding than tank-style
- Eliminates need for rotation actions
- Good for grid-aligned environments

### 8-Way Movement
- Larger action space (8 directions)
- Most efficient pathfinding (diagonal shortcuts)
- Best for open environments with fewer obstacles
- Highest flexibility but more complex action space

### Tank-Style Movement
- Smallest individual action spaces (2 for move, 4 for rotate)
- Requires action sequences for diagonal movement
- Natural for environments with facing-dependent mechanics
- Best for combat scenarios where orientation matters

## Training Tips

### Behavioral Differences
- **Efficiency**: Direct movement can be more efficient as it eliminates rotation actions
- **Exploration**: Random action policies behave differently - cardinal/8-way move in all directions equally rather than mostly forward/backward
- **Strategies**: Policies may need different strategies that don't rely on maintaining specific orientations

### Best Practices
1. **Curriculum Learning**: Start with simple navigation tasks before complex scenarios
2. **Reward Shaping**: Consider rewards that encourage efficient pathfinding
3. **Action Masking**: Mask invalid directions (e.g., into walls) to speed up learning
4. **Consistent Configuration**: Always use the same movement configuration for training and evaluation

## Important Notes

1. **Orientation Behavior**: 
   - Tank movement changes agent orientation with rotate action
   - Cardinal and 8-way movement do NOT change orientation
   - Attack action always uses the agent's current facing direction

2. **Training Compatibility**: 
   - Use `++trainer.env_overrides` (double plus) to force override action settings during training
   - Use `+replay_job.sim.env_overrides` (single plus) to add overrides during play/eval
   - Policies trained with one movement type won't work well with different movement types

3. **Device Configuration**: 
   - Always specify `device=cpu` on macOS to avoid MPS issues
   - Remove this parameter when running on machines with CUDA GPUs

4. **Checkpoints**: 
   - The action space dimensions must match between training and evaluation
   - Store your movement configuration with your checkpoints for reproducibility

## Backward Compatibility

The system maintains full backward compatibility:
- Existing configurations work unchanged (tank-style movement is the default)
- New movement actions are disabled by default
- Policies trained before these changes continue to work exactly as before