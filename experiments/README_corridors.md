# Corridor Generation System

This is the unified corridor generation system for Metta, providing a simple and powerful API for creating any type of
corridor pattern.

## Core Concept

All corridors are created using a single unified function that takes an angle:

- **0°** = East (right)
- **90°** = North (up)
- **180°** = West (left)
- **270°** = South (down)
- **Any angle** works!

## Main Files

### Scene Implementation

- `metta/map/scenes/angled_corridor_builder.py` - The unified corridor builder that handles all angles

### Navigation Integration

- `experiments/evals/navigation_with_corridors.py` - Navigation eval suite using procedural corridors
- `experiments/test_navigation_corridors.py` - Test script for navigation integration
- `experiments/visualize_navigation_maps.py` - Full visualization tool with plots
- `experiments/quick_map_check.py` - Quick ASCII visualization of generated maps

### Examples & Tests

- `experiments/corridor_examples.py` - Consolidated examples showing all use cases
- `experiments/test_radial_corridors.py` - Comprehensive tests for radial patterns
- `experiments/simple_radial_example.py` - Simple examples for radial patterns

### Supporting Systems

- `metta/map/metrics.py` - Map metrics computation for analyzing generated maps

## Quick Start

### Basic Corridor

```python
from metta.map.scenes.angled_corridor_builder import corridor

# Create a corridor at any angle
corridor(center=(15, 15), angle=45, length=20, thickness=3)
```

### Convenience Functions

```python
from metta.map.scenes.angled_corridor_builder import horizontal, vertical

# Horizontal corridor
horizontal(y=10, thickness=5, x_start=5, x_end=45)

# Vertical corridor
vertical(x=20, thickness=3, y_start=5, y_end=35)
```

### Radial Pattern (like radial_mini.map)

```python
from metta.map.scenes.angled_corridor_builder import radial_corridors

corridors = radial_corridors(
    center=(11, 11),
    num_spokes=8,
    length=8,
    thickness=1
)
```

### Complex Layout (like corridors.map)

```python
corridors = [
    horizontal(y=13, thickness=7),  # Thick horizontal band
    vertical(x=10, thickness=3),     # Multiple vertical corridors
    vertical(x=20, thickness=4),
    vertical(x=30, thickness=2),
    # ... etc
]
```

## Key Features

1. **Unified API** - One function handles all angles (0-360°)
2. **Perfect Symmetry** - Radial patterns are perfectly symmetrical
3. **No Gaps** - Diagonal corridors are continuous and traversable
4. **Variable Thickness** - Works correctly for any thickness at any angle
5. **Bidirectional** - Can extend corridors in both directions from center
6. **Object Placement** - Smart placement at ends, centers, or intersections

## Navigation Integration

The corridor system is directly integrated into the navigation evaluation suite:

```python
from experiments.evals.navigation_with_corridors import (
    make_corridors_env,      # Generates corridors.map pattern
    make_radial_mini_env,    # Generates radial_mini.map pattern
    make_radial_small_env,   # Generates radial_small.map pattern
    make_radial_large_env,   # Generates radial_large.map pattern
    make_grid_maze_env,      # Generates grid/maze pattern
)
```

This replaces static ASCII map files with dynamically generated corridor patterns, allowing for:

- Consistent map generation
- Easy parameter tweaking
- Procedural variation
- Reduced file dependencies

## Visualization

### Quick ASCII Check

```bash
python experiments/quick_map_check.py
```

Shows ASCII representation of all generated maps in the terminal.

### Full Visualization with Plots

```bash
# View all maps
python experiments/visualize_navigation_maps.py

# View specific map
python experiments/visualize_navigation_maps.py --map corridors

# Save plots to files
python experiments/visualize_navigation_maps.py --save

# Compare with original ASCII maps
python experiments/visualize_navigation_maps.py --compare
```

### In-Code Visualization

```python
from experiments.evals.navigation_with_corridors import visualize_env_map, make_corridors_env

# Quick ASCII visualization
grid = visualize_env_map(make_corridors_env, title="Corridors Map")
```

## Examples

See `experiments/corridor_examples.py` for complete examples of:

- Recreating corridors.map
- Recreating radial_mini.map
- Custom angled corridors
- Star patterns
- Grid patterns

## Clean Architecture

The final implementation is clean and modular:

- Single scene class handles all corridor types
- No duplicate functionality
- Clear, intuitive API
- Comprehensive examples and tests
