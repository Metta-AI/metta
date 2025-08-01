# Movement Types in Metta

This document describes the three movement systems available in Metta and provides commands for training and playing with each type.

## Movement Types Overview

### 1. Tank Movement (Default)
- **Actions**: `move` (forward/backward) + `rotate` (turn left/right)
- **Behavior**: Agents move in the direction they're facing, must rotate to change direction
- **Use Case**: Classic tank-style controls, good for combat scenarios where facing matters

### 2. Cardinal Movement
- **Actions**: `move_cardinal` (4 directions: N, E, S, W)
- **Behavior**: Direct movement in 4 cardinal directions, no rotation needed
- **Use Case**: Simplified movement for navigation tasks

### 3. 8-Way Movement
- **Actions**: `move_8way` (8 directions: N, NE, E, SE, S, SW, W, NW)
- **Behavior**: Direct movement in 8 directions including diagonals, no rotation needed
- **Use Case**: Most flexible movement for complex navigation

## Training Commands

### Tank Movement Training
```bash
./tools/train.py run=tank_movement_test +hardware=macbook \
  trainer.curriculum=/env/mettagrid/navigation/training/small \
  trainer.total_timesteps=500000 \
  trainer.num_workers=2
```

### Cardinal Movement Training
```bash
./tools/train.py run=cardinal_movement_test +hardware=macbook \
  trainer.curriculum=/env/mettagrid/navigation/training/small \
  ++trainer.env_overrides.game.actions.move.enabled=false \
  ++trainer.env_overrides.game.actions.rotate.enabled=false \
  ++trainer.env_overrides.game.actions.move_cardinal.enabled=true \
  trainer.total_timesteps=500000 \
  trainer.num_workers=2
```

### 8-Way Movement Training
```bash
./tools/train.py run=8way_movement_test +hardware=macbook \
  trainer.curriculum=/env/mettagrid/navigation/training/small \
  ++trainer.env_overrides.game.actions.move.enabled=false \
  ++trainer.env_overrides.game.actions.rotate.enabled=false \
  ++trainer.env_overrides.game.actions.move_8way.enabled=true \
  trainer.total_timesteps=500000 \
  trainer.num_workers=2
```

## Play/Evaluation Commands

### Tank Movement Play
```bash
./tools/play.py run=play_tank \
  policy_uri=file://./train_dir/tank_movement_test/checkpoints \
  replay_job.sim.env=/env/mettagrid/arena/basic \
  +hardware=macbook \
  device=cpu
```

### Cardinal Movement Play
```bash
./tools/play.py run=play_cardinal \
  policy_uri=file://./train_dir/cardinal_movement_test/checkpoints \
  replay_job.sim.env=/env/mettagrid/arena/basic \
  +replay_job.sim.env_overrides.game.actions.move.enabled=false \
  +replay_job.sim.env_overrides.game.actions.rotate.enabled=false \
  +replay_job.sim.env_overrides.game.actions.move_cardinal.enabled=true \
  +hardware=macbook \
  device=cpu
```

### 8-Way Movement Play
```bash
./tools/play.py run=play_8way \
  policy_uri=file://./train_dir/8way_movement_test/checkpoints \
  replay_job.sim.env=/env/mettagrid/arena/basic \
  +replay_job.sim.env_overrides.game.actions.move.enabled=false \
  +replay_job.sim.env_overrides.game.actions.rotate.enabled=false \
  +replay_job.sim.env_overrides.game.actions.move_8way.enabled=true \
  +hardware=macbook \
  device=cpu
```

## Using Pre-configured Environments

You can also use the example configuration files:

### Cardinal Movement Example
```bash
# Training
./tools/train.py run=cardinal_example +hardware=macbook \
  trainer.env=/env/mettagrid/cardinal_movement_example \
  trainer.total_timesteps=500000

# Playing
./tools/play.py run=play_cardinal_example \
  policy_uri=file://./train_dir/cardinal_example/checkpoints \
  replay_job.sim.env=/env/mettagrid/cardinal_movement_example \
  +hardware=macbook device=cpu
```

### 8-Way Movement Example
```bash
# Training
./tools/train.py run=8way_example +hardware=macbook \
  trainer.env=/env/mettagrid/8way_movement_example \
  trainer.total_timesteps=500000

# Playing
./tools/play.py run=play_8way_example \
  policy_uri=file://./train_dir/8way_example/checkpoints \
  replay_job.sim.env=/env/mettagrid/8way_movement_example \
  +hardware=macbook device=cpu
```

## Movement Configuration Details

### Action Space Changes

Each movement type modifies the action space:

| Movement Type | Actions Enabled | Action Space Size |
|--------------|----------------|-------------------|
| Tank | move, rotate, noop, attack, etc. | Varies by config |
| Cardinal | move_cardinal, noop, attack, etc. | 4 directions + other actions |
| 8-Way | move_8way, noop, attack, etc. | 8 directions + other actions |

### Important Notes

1. **Orientation**: 
   - Tank movement changes agent orientation with rotate action
   - Cardinal and 8-way movement do NOT change orientation
   - Attack action always uses the agent's current facing direction

2. **Training**: 
   - Use `++trainer.env_overrides` (double plus) to force override action settings during training
   - Use `+replay_job.sim.env_overrides` (single plus) to add overrides during play/eval

3. **Device**: 
   - Always specify `device=cpu` on macOS to avoid MPS issues
   - Remove this parameter when running on machines with CUDA GPUs

4. **Checkpoints**: 
   - Policies trained with one movement type won't work well with different movement types
   - The action space dimensions must match between training and evaluation