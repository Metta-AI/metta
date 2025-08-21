# Movement Systems in Metta

Metta supports three movement systems configurable through command-line overrides:

## Movement Types

### 1. Tank Movement (Default)
- **Actions**: `move` (0=forward, 1=backward), `rotate` (0-3=Up/Down/Left/Right)
- **Behavior**: Move relative to facing direction, must rotate to change direction
- **Use Case**: Combat scenarios where facing matters

### 2. Cardinal Movement
- **Actions**: `move_cardinal` (0=North, 1=South, 2=West, 3=East)
- **Behavior**: Direct 4-directional movement that updates orientation to match movement direction
- **Use Case**: Navigation tasks, grid-aligned environments

### 3. 8-Way Movement
- **Actions**: `move_8way` (0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW)
- **Behavior**: Direct 8-directional movement including diagonals; when there is an object in the way of a diagonal move, default to moving in
adjacent cardinal direction pointing towards the target square
- **Use Case**: Complex navigation, open environments

## Key Concepts

**Orientation**: Both `rotate` and direct movement actions (`move_cardinal`, `move_8way`) change agent orientation. With direct movement, orientation is automatically updated to match the movement direction. Orientation affects actions like `attack`.

**Action Space**: Enabled actions determine the action space indices:
```python
# Tank only: ["noop", "move", "rotate", "attack"]
# Cardinal only: ["noop", "move_cardinal", "attack"]
# Hybrid: ["noop", "move", "rotate", "move_cardinal", "attack"]
```

## Usage Examples

### Training
```bash
# Tank (default - no overrides needed)
./tools/train.py run=tank_test trainer.total_timesteps=500000

# Cardinal
./tools/train.py run=cardinal_test \
  ++trainer.env_overrides.game.actions.move.enabled=false \
  ++trainer.env_overrides.game.actions.rotate.enabled=false \
  ++trainer.env_overrides.game.actions.move_cardinal.enabled=true

# 8-Way
./tools/train.py run=8way_test \
  ++trainer.env_overrides.game.actions.move.enabled=false \
  ++trainer.env_overrides.game.actions.rotate.enabled=false \
  ++trainer.env_overrides.game.actions.move_8way.enabled=true
```

### Evaluation
```bash
# Tank (default - no overrides needed)
./tools/play.py run=play_tank \
  policy_uri=file://./train_dir/tank_test/checkpoints

# Cardinal
./tools/play.py run=play_cardinal \
  policy_uri=file://./train_dir/cardinal_test/checkpoints \
  +replay_job.sim.env_overrides.game.actions.move.enabled=false \
  +replay_job.sim.env_overrides.game.actions.rotate.enabled=false \
  +replay_job.sim.env_overrides.game.actions.move_cardinal.enabled=true

# 8-Way
./tools/play.py run=play_8way \
  policy_uri=file://./train_dir/8way_test/checkpoints \
  +replay_job.sim.env_overrides.game.actions.move.enabled=false \
  +replay_job.sim.env_overrides.game.actions.rotate.enabled=false \
  +replay_job.sim.env_overrides.game.actions.move_8way.enabled=true
```

## Performance Trade-offs

| Movement Type | Action Space | Pathfinding | Best For |
|--------------|--------------|-------------|----------|
| Tank | Small (2+4) | Requires rotation | Combat, facing-dependent mechanics |
| Cardinal | Medium (4) | Direct | Navigation, grid environments |
| 8-Way | Large (8) | Most efficient | Open environments, complex navigation |

## Important Notes

- **Override Syntax**: Use `++` for training (force override), `+` for evaluation (add override)
- **Compatibility**: Policies must be evaluated with the same movement type they were trained with
- **macOS**: Add `device=cpu` to avoid MPS issues
- **Backward Compatibility**: Tank movement is default; existing configs work unchanged
