# Map Generation System

The Metta map generation system provides a flexible, hierarchical approach to creating diverse environments for multi-agent reinforcement learning experiments.

## Overview

Maps in Metta are generated using a **scene-based system** that allows for modular, composable map construction. The system supports both procedural generation and predefined patterns, making it suitable for a wide variety of experimental setups.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   YAML Config   │────▶│     MapGen       │────▶│   Grid Output   │
│  (map_builder)  │     │ (LevelBuilder)   │     │  (numpy array)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │   Scene Tree     │
                        │ (Hierarchical)   │
                        └──────────────────┘
```

## Core Components

### 1. MapGen (Main Builder)

Located in `metta/map/mapgen.py`, this is the primary map builder:

```python
class MapGen(LevelBuilder):
    def __init__(self, **kwargs):
        # Configures width, height, border_width, root scene

    def build(self) -> Level:
        # Generates the map grid and metadata
```

### 2. Scene System

Scenes are modular components that generate different parts of maps. Each scene:
- Inherits from `Scene[ParamsT]`
- Has typed parameters via a `Config` class
- Can have child scenes for hierarchical generation
- Implements a `render()` method

### 3. Grid System

Maps are represented as 2D numpy arrays where each cell contains a string identifier (e.g., "wall", "empty", "agent_0").

## Available Scene Types

### Core Scenes

| Scene | Purpose | Key Parameters |
|-------|---------|----------------|
| `Auto` | Automated map generation with BSP/grid layouts | `layout`, `room_objects`, `content` |
| `BSP` | Binary space partitioning for rooms | `area_count`, `min_area_size` |
| `RoomGrid` | Rectangular grid of rooms | `rows`, `columns`, `border_width` |
| `Maze` | Procedural maze generation | `room_size`, `wall_size` |
| `Random` | Random object/agent placement | `agents`, `objects` |
| `WFC` | Wave Function Collapse pattern generation | `pattern`, `pattern_size` |
| `Mirror` | Symmetry operations | `axes` (horizontal/vertical/x4) |
| `Layout` | Custom area definitions | `areas` with positions/tags |

### Utility Scenes

- `MakeConnected`: Ensures all areas are reachable
- `RandomObjects`: Places objects based on density distributions
- `RandomScene`: Randomly selects from scene options
- `RemoveAgents`: Removes agents from specific areas

## Configuration

### Basic Map Configuration

```yaml
game:
  map_builder:
    _target_: metta.map.mapgen.MapGen
    width: 120
    height: 120
    border_width: 5  # Default: prevents agents seeing beyond walls

    root:
      type: metta.map.scenes.auto.Auto
      params:
        num_agents: 24
```

### Scene Definition Methods

#### 1. Inline Definition

Define scenes directly in your configuration:

```yaml
root:
  type: metta.map.scenes.maze.Maze
  params:
    room_size: ["uniform", 1, 3]
    wall_size: ["uniform", 1, 3]
  children:
    - where: "full"
      scene:
        type: metta.map.scenes.random.Random
        params:
          agents: 4
```

#### 2. External Scene Files

Reference predefined scenes from the `scenes/` directory:

```yaml
content:
  - scene: /wfc/dungeons.yaml
    weight: 20
  - scene: /wfc/mazelike1.yaml
    weight: 10
```

External scene files follow the same structure:

```yaml
# scenes/wfc/dungeons.yaml
type: metta.map.scenes.wfc.WFC
params:
  pattern_size: 3
  pattern: |-
    ###.....####
    ###.....####
    ...........#
```

### Distribution Parameters

Many scenes support distribution-based parameters for randomization:

```yaml
objects:
  mine_red: ["uniform", 0.001, 0.01]      # Uniform distribution
  altar: ["lognormal", 0.0001, 0.01, 0.03] # Log-normal distribution
  wall: 5                                   # Fixed count
```

## Usage Examples

### Simple Random Map

```yaml
map_builder:
  _target_: metta.map.mapgen.MapGen
  width: 50
  height: 50
  root:
    type: metta.map.scenes.random.Random
    params:
      agents: 4
      objects:
        wall: 20
        altar: 2
```

### Multi-Room Map with Patterns

```yaml
map_builder:
  _target_: metta.map.mapgen.MapGen
  width: 100
  height: 100
  root:
    type: metta.map.scenes.room_grid.RoomGrid
    params:
      rows: 2
      columns: 2
      border_width: 3
    children:
      - where:
          tags: ["room"]
        scene: /wfc/dungeons.yaml
```

### Procedural Map with Auto Scene

```yaml
map_builder:
  _target_: metta.map.mapgen.MapGen
  root:
    type: metta.map.scenes.auto.Auto
    params:
      num_agents: 16
      layout:
        grid: 1
        bsp: 1
      grid:
        rows: ["uniform", 2, 4]
        columns: ["uniform", 2, 4]
      room_objects:
        mine_red: ["uniform", 0.001, 0.01]
        generator_red: ["lognormal", 0.0001, 0.01, 0.03]
      content:
        - scene: /wfc/dungeons.yaml
          weight: 3
        - scene:
            type: metta.map.scenes.maze.Maze
          weight: 1
```

## Creating Custom Scenes

To create a custom scene:

1. **Define Parameter Class**:
```python
from metta.common.util.config import Config

class MySceneParams(Config):
    my_param: int = 10
    another_param: str = "default"
```

2. **Implement Scene Class**:
```python
from metta.map.scene import Scene

class MyScene(Scene[MySceneParams]):
    def render(self):
        # Access grid via self.area.grid
        # Access parameters via self.params.my_param
        # Use self.rng for randomization
        pass

    def get_children(self) -> list[ChildrenAction]:
        # Optional: define child scenes
        return []
```

3. **Use in Configuration**:
```yaml
root:
  type: my_module.MyScene
  params:
    my_param: 20
    another_param: "custom"
```

## Viewing and Debugging Maps

### Command Line Tools

```bash
# Generate and view a single map
./tools/map/gen.py configs/env/mettagrid/map_builder/auto.yaml

# Generate multiple maps
./tools/map/gen.py --count=10 configs/env/mettagrid/map_builder/auto.yaml

# Save to S3
./tools/map/gen.py --output-uri=s3://bucket/maps/ configs/env/mettagrid/map_builder/auto.yaml

# View existing map
./tools/map/view.py s3://bucket/maps/map_001.yaml
```

### Programmatic Generation

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Load config
cfg = OmegaConf.load("config.yaml")
map_builder = instantiate(cfg.game.map_builder)

# Generate map
level = map_builder.build()

# Access grid
grid = level.grid  # numpy array
labels = level.labels  # metadata
```

## Best Practices

1. **Use External Scenes** for reusable patterns
2. **Leverage Distributions** for variety in procedural generation
3. **Compose Simple Scenes** to create complex layouts
4. **Test Maps Visually** before training
5. **Version Control Scene Files** for reproducibility

## Integration with Curriculum Learning

Maps integrate seamlessly with the curriculum system through environment configurations. See the [Curriculum README](../mettagrid/src/metta/mettagrid/curriculum/README.md) for details on how maps are used in learning curricula.
