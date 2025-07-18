# Movement Metrics

Tracks agent navigation behavior with minimal overhead when disabled.

## Metrics

- `movement/facing/[up|down|left|right]` - Time spent facing each direction
- `movement/sequential_rotations` - Sequential rotation sequences (indicates search/indecision)

## Enabling

```yaml
# Config file
game:
  track_movement_metrics: true
```

```python
# API
env = Environment(track_movement_metrics=True)
```

```bash
# Training
./tools/train.py ... +trainer.env_overrides.game.track_movement_metrics=true

# Navigation recipe
MOVEMENT_METRICS=true ./recipes/navigation.sh
```

## Performance

- **Disabled by default** (zero overhead)
- **~10% overhead when enabled** in GPU training
- Recommended for debugging/analysis only, not full training runs

## WandB

Metrics appear as `env_agent/movement/*` with full statistical aggregations.

## Tests

Located in `mettagrid/tests/`:
- `test_movement_metrics.py` - Basic functionality
- `test_movement_metrics_performance.py` - Performance measurement
- `test_movement_metrics_api.py` - API integration
