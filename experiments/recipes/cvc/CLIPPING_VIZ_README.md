# Clipping Visualization Recipe

This recipe helps visualize and validate the clipping system's behavior, particularly useful for testing the new bit-shifting implementation.

## Quick Start

Run the default clipping visualization:

```bash
uv run ./tools/run.py experiments.recipes.cvc.clipping_viz.play
```

This will:
- Start a 50x50 map with 4 agents
- Begin with hub stations already clipped
- Clip buildings every ~20 steps
- Use auto-calculated `length_scale` based on percolation theory
- Use `scaled_cutoff_distance=3` (buildings beyond 3× length_scale are unaffected)

## Variants

### Slow Clipping (for careful observation)
```bash
uv run ./tools/run.py experiments.recipes.cvc.clipping_viz.play_slow
```
- Clips every ~50 steps instead of ~20
- Better for observing the spatial diffusion pattern

### Aggressive Clipping (for stress testing)
```bash
uv run ./tools/run.py experiments.recipes.cvc.clipping_viz.play_aggressive
```
- Clips every ~10 steps
- Uses `scaled_cutoff_distance=2` for more localized spreading
- Good for testing edge cases

### Large Map
```bash
uv run ./tools/run.py experiments.recipes.cvc.clipping_viz.play_large_map
```
- Uses 70x70 map with 8 agents
- Tests clipping over larger spatial scales

### Custom Parameters
```bash
uv run ./tools/run.py experiments.recipes.cvc.clipping_viz.play_custom \
    clip_period=15 \
    length_scale=10.0 \
    scaled_cutoff_distance=4 \
    mission=extractor_hub_70
```

## What to Look For

### Visual Validation

1. **Clipping Spread Pattern**
   - Buildings closer to clipped buildings should clip more frequently
   - Buildings far away should remain mostly unaffected
   - The spread should appear "organic" and spatially coherent

2. **Infection Weights** (with the new bit-shifting)
   - Buildings at distance `d` get weight: `1 << (scaled_cutoff_distance - scaled_distance)`
   - where `scaled_distance = d / length_scale`
   - Buildings beyond `scaled_cutoff_distance * length_scale` get weight 0

3. **Spatial Cutoff**
   - Buildings beyond the cutoff should almost never clip (unless they get independently clipped)
   - The transition should be sharp at the cutoff boundary

### Parameter Tuning

**`clip_period`** (default: 0, meaning disabled)
- Controls how often clipping events occur
- Lower = more frequent clipping
- In this recipe: 10-50 range is good for visualization

**`length_scale`** (default: 0.0 = auto-calculate)
- Controls spatial spread rate
- Larger values = clipping spreads further
- Auto-calculation uses percolation theory: `(grid_size / sqrt(num_buildings)) * sqrt(4.51 / 4π)`
- For 50x50 maps with ~25 buildings, this calculates to ~6

**`scaled_cutoff_distance`** (default: 3)
- Maximum distance in units of `length_scale`
- Buildings beyond `cutoff_distance * length_scale` get zero infection weight
- Lower values = more localized clipping
- Default of 3 means buildings beyond ~18 grid units (3 × 6) are unaffected

## New Bit-Shifting Implementation

The branch updates clipping from exponential to bit-shifting:

**Old:** `weight = exp(-distance / length_scale) * 1000000`

**New:** `weight = 1 << (scaled_cutoff_distance - scaled_distance)`

Where `scaled_distance = floor(distance / length_scale)`

### Why Bit Shifting?

1. **Performance**: Bit shifting is much faster than computing exponentials
2. **Precision**: Integer arithmetic avoids floating-point errors
3. **Intuitive**: Powers of 2 make relative weights easy to reason about

### Example Weights (with `scaled_cutoff_distance=3`)

| Distance | Scaled Distance | Weight (binary) | Weight (decimal) |
|----------|-----------------|-----------------|------------------|
| 0        | 0               | 1 << 3          | 8                |
| 1×LS     | 1               | 1 << 2          | 4                |
| 2×LS     | 2               | 1 << 1          | 2                |
| 3×LS     | 3               | 1 << 0          | 1                |
| >3×LS    | >3              | 0               | 0                |

Compare to old exponential (at same distances):
- `exp(0)` ≈ 1.0
- `exp(-1)` ≈ 0.368
- `exp(-2)` ≈ 0.135
- `exp(-3)` ≈ 0.050

The bit-shifting provides cleaner 2× drop-offs rather than exponential decay.

## Troubleshooting

**Clipping isn't happening:**
- Check that `clip_period > 0`
- Ensure at least one building starts clipped (`start_hub_clipped=True`)

**Clipping spreads too far:**
- Reduce `scaled_cutoff_distance`
- Increase `length_scale` (or let it auto-calculate)

**Clipping is too localized:**
- Increase `scaled_cutoff_distance`
- Manually set a smaller `length_scale`

**Can't see the pattern:**
- Use `play_slow` for slower clipping
- Try a larger map with `play_large_map`

