# Perfect Maze Configurations

Perfect mazes are mazes where there is exactly one path between any two points, with no loops or inaccessible areas. This collection provides various configurations for generating perfect mazes using different algorithms and layouts.

## Available Configurations

### `perfect_maze_basic.yaml`
- **Purpose**: Minimal perfect maze configuration for basic usage
- **Size**: 21x21 (standard maze size)
- **Features**: No agents or special objects, pure maze generation
- **Algorithm**: Default (recursive_backtracking)

### `perfect_maze.yaml`
- **Purpose**: Complete perfect maze with agent and goal
- **Size**: 21x21
- **Features**:
  - Agent spawns at entrance (top-left)
  - Goal placed at exit (bottom-right)
  - Uses recursive backtracking algorithm
- **Use case**: Single-player maze solving scenarios

### `perfect_maze_small.yaml`
- **Purpose**: Small maze for quick testing and development
- **Size**: 11x11
- **Features**:
  - Faster generation for rapid iteration
  - Uses Prim's algorithm for organic feel
  - Agent and goal placement
- **Use case**: Development and debugging

### `perfect_maze_large.yaml`
- **Purpose**: Large-scale maze with multiple instances
- **Size**: 31x31 with 4 instances
- **Features**:
  - 4 separate maze instances with 5-tile borders
  - Uses Kruskal's algorithm for balanced structure
  - Agents and goals in each instance
- **Use case**: Multi-agent environments or parallel training

### `perfect_maze_algorithms.yaml`
- **Purpose**: Algorithm comparison and demonstration
- **Size**: 60x20 (3 mazes side by side)
- **Features**:
  - Shows all three algorithms in one layout
  - Numbered markers (1, 2, 3) to identify each algorithm
  - Great for visual comparison of algorithm characteristics
- **Use case**: Educational purposes and algorithm selection

## Algorithms

### Recursive Backtracking (Default)
- **Characteristics**: Long, winding corridors with deep branching
- **Pattern**: Creates "river-like" main passages
- **Best for**: Traditional maze feel, longer solution paths

### Kruskal's Algorithm
- **Characteristics**: More balanced tree structure, shorter dead ends
- **Pattern**: Evenly distributed branching
- **Best for**: Balanced difficulty, multi-agent scenarios

### Prim's Algorithm
- **Characteristics**: Organic growth from center, flowing shapes
- **Pattern**: Radial expansion with natural curves
- **Best for**: Aesthetically pleasing mazes, exploration scenarios

## Usage Examples

### Command Line Generation
```bash
# Basic maze
./tools/map/gen.py configs/env/mettagrid/game/map_builder/perfect_maze_basic.yaml

# With visualization
./tools/map/gen.py --show-mode=mettascope configs/env/mettagrid/game/map_builder/perfect_maze.yaml

# Generate multiple mazes
./tools/map/gen.py --count=10 configs/env/mettagrid/game/map_builder/perfect_maze_small.yaml
```

### Python Integration
```python
from metta.map.mapgen import MapGen
from omegaconf import OmegaConf

# Load configuration
cfg = OmegaConf.load("configs/env/mettagrid/game/map_builder/perfect_maze.yaml")
mapgen = MapGen(**cfg)

# Generate maze
level = mapgen.build()
maze_grid = level.grid
```

### Customization
You can customize any configuration by overriding parameters:

```yaml
# Override algorithm and size
_target_: metta.map.mapgen.MapGen
width: 25
height: 25

root:
  type: metta.map.scenes.perfect_maze.PerfectMaze
  params:
    algorithm: kruskal  # Change algorithm
    entrance: bottom-left  # Change entrance location
    exit: top-right       # Change exit location
```

## Perfect Maze Properties

- ✅ **Connectivity**: Every cell is reachable from every other cell
- ✅ **Uniqueness**: Exactly one path between any two points
- ✅ **No loops**: Tree structure with no cycles
- ✅ **No inaccessible areas**: All maze cells are part of the solution space
- ✅ **One-tile corridors**: Classic maze appearance with single-width passages

## Tips

1. **Odd dimensions recommended**: Use odd width/height (e.g., 21, 31, 41) for best maze structure
2. **Algorithm selection**:
   - Recursive backtracking for traditional feel
   - Kruskal for balanced difficulty
   - Prim for organic appearance
3. **Performance**: Smaller mazes (≤15x15) generate very quickly, larger ones may take more time
4. **Entrance/Exit**: Optional parameters - omit for corner-only accessibility
