# Clipper Percolation Length Scale - Quick Start

## Overview

The clipper's `length_scale` is **automatically calculated** based on percolation theory when you create a `GameConfig` with a clipper. No extra configuration needed!

## Formula

```python
THEORETICAL_LENGTH_SCALE = (GRID_SIZE / √NUM_BUILDINGS) × √(4.51 / 4π) × fudge_factor
```

Where:
- **4.51** = Critical percolation threshold for 2D continuum percolation
- **GRID_SIZE** = max(width, height) from map_builder
- **NUM_BUILDINGS** = Number of assemblers in the environment
- **fudge_factor** = `length_scale_factor` (default=1.0)

## Quick Usage

### Default (Automatic)

```python
from mettagrid.config.mettagrid_config import GameConfig, ClipperConfig
from mettagrid.map_builder.random import RandomMapBuilder

game_config = GameConfig(
    map_builder=RandomMapBuilder.Config(width=50, height=50),
    objects={
        f"assembler_{i}": AssemblerConfig(...)
        for i in range(25)
    },
    clipper=ClipperConfig(
        recipe=RecipeConfig(input_resources={"ore_red": 5}),
        clip_rate=0.1,
        # That's it! length_scale is calculated automatically
    ),
)

print(game_config.clipper.length_scale)  # ~5.991
```

### With Fudge Factor

Tune the percolation calculation with `length_scale_factor`:

```python
clipper=ClipperConfig(
    recipe=RecipeConfig(input_resources={"ore_red": 5}),
    clip_rate=0.1,
    length_scale_factor=1.5,  # Fudge factor for tuning
)
```

### Manual Override

Disable auto-calculation and use a manual value:

```python
clipper=ClipperConfig(
    recipe=RecipeConfig(input_resources={"ore_red": 5}),
    clip_rate=0.1,
    auto_length_scale=False,  # Disable automatic calculation
    length_scale=3.14,  # Use manual value
)
```

## Parameters

### `auto_length_scale` (default=True)
- **True**: Automatically calculate from percolation theory
- **False**: Use manual `length_scale` value

### `length_scale_factor` (default=1.0)
- Fudge factor for tuning the automatic calculation
- **< 1.0**: More localized clipping
- **= 1.0**: Theoretical percolation threshold
- **> 1.0**: More diffuse clipping

### `length_scale`
- Automatically set when `auto_length_scale=True`
- Manually set when `auto_length_scale=False`
- Controls spatial spread: `weight = exp(-distance / length_scale)`

## Example Values

### 50×50 grid with 25 buildings:

| Fudge Factor | Length Scale |
|--------------|--------------|
| 0.5          | 2.995        |
| 1.0          | 5.991        |
| 1.5          | 8.986        |
| 2.0          | 11.982       |

### 50×50 grid with varying building counts (fudge=1.0):

| Buildings | Length Scale |
|-----------|--------------|
| 10        | 9.472        |
| 25        | 5.991        |
| 50        | 4.236        |
| 100       | 2.995        |

## See Also

- Example code: `examples/clipper_auto_percolation.py`
- Tests: `tests/test_clipper_percolation.py`
