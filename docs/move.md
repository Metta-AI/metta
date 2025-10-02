# Movement System in Metta

Metta uses a unified movement system with a single `move` action that can be configured for either 4-way (cardinal) or
8-way (diagonal) movement.

## Movement Configuration

### Unified Move Action

- **Action**: `move` (directions depend on `allow_diagonals` flag)
- **Cardinal Mode**: `allow_diagonals=false` → 4 directions (0=N, 1=S, 2=W, 3=E)
- **8-Way Mode**: `allow_diagonals=true` → 8 directions (0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW)
- **Behavior**: Direct movement that updates orientation to match movement direction

## Key Concepts

**Orientation**: The `move` action automatically updates agent orientation to match the movement direction. Orientation
affects actions like `attack`.

**Action Space**: The `move` action's parameter count depends on the `allow_diagonals` setting:

```python
# Cardinal (4-way): move action takes values 0-3
# 8-Way (diagonal): move action takes values 0-7
```

## Usage Examples

### Training

```bash
# Cardinal Movement (4-way, default)
uv run ./tools/run.py train arena run=cardinal_test \
  ++trainer.env_overrides.game.actions.move.enabled=true \
  ++trainer.env_overrides.game.allow_diagonals=false

# 8-Way Movement (with diagonals)
uv run ./tools/run.py train arena run=8way_test \
  ++trainer.env_overrides.game.actions.move.enabled=true \
  ++trainer.env_overrides.game.allow_diagonals=true
```

### Evaluation

```bash
# Cardinal Movement
uv run ./tools/run.py evaluate arena \
  policy_uri=file://./train_dir/cardinal_test/checkpoints/cardinal_test:v12.pt \
  +replay_job.sim.env_overrides.game.actions.move.enabled=true \
  +replay_job.sim.env_overrides.game.allow_diagonals=false

# 8-Way Movement
uv run ./tools/run.py evaluate arena \
  policy_uri=file://./train_dir/8way_test/checkpoints/8way_test:v12.pt \
  +replay_job.sim.env_overrides.game.actions.move.enabled=true \
  +replay_job.sim.env_overrides.game.allow_diagonals=true
```

## Performance Trade-offs

| Movement Type | Action Space | Configuration                        | Best For                              |
| ------------- | ------------ | ------------------------------------ | ------------------------------------- |
| Cardinal      | Medium (4)   | `move=true`, `allow_diagonals=false` | Navigation, grid environments         |
| 8-Way         | Large (8)    | `move=true`, `allow_diagonals=true`  | Open environments, complex navigation |

## Important Notes

- **Diagonal Control**: Use `allow_diagonals` flag to control whether `move` supports diagonal directions
- **Override Syntax**: Use `++` for training (force override), `+` for evaluation (add override)
- **Compatibility**: Policies must be evaluated with the same movement configuration they were trained with
- **macOS**: Add `device=cpu` to avoid MPS issues
- **Migration**: Old `move_8way` and `move_cardinal` actions are deprecated; use `move` with `allow_diagonals`
