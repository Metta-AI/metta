# Navigation Curriculum in Metta AI

The **navigation curriculum** is the other primary training environment in Metta AI, focusing on spatial navigation, pathfinding, and exploration skills in gridworld environments. Unlike the arena curriculum which emphasizes multi-agent cooperation and competition, the navigation curriculum is designed to develop fundamental navigation intelligence.

## What is the Navigation Curriculum?

The navigation curriculum trains agents to navigate complex gridworld environments, find goals (altars), and optimize their movement strategies. It's a curriculum learning system that progressively increases navigation difficulty through varied terrain types and map configurations.

## Key Components:

### 1. **Navigation Environment**
- **Grid-based navigation**: Agents must find and reach altar objects to score points
- **Single agent focus**: Typically 4 agents per environment (vs 24 in arena)
- **Limited actions**: Only move, rotate, and get_items actions available
- **Resource system**: Agents collect "heart" rewards from altars
- **Energy management**: Actions cost energy, requiring efficient pathfinding

### 2. **Curriculum Learning System**
The navigation curriculum uses a sophisticated **multi-bucket task generation** system with two main task types:

#### Dense Terrain Tasks
- Uses pre-generated terrain maps from varied terrain directories
- **Terrain types**: balanced, maze, sparse, dense, cylinder-world
- **Sizes**: large, medium, small
- **Altar counts**: Variable (3-50 altars per map)
- **Special maps**: terrain_maps_nohearts (navigation without rewards)

#### Sparse Terrain Tasks
- Procedurally generated maps using RandomMapBuilder
- **Map sizes**: 60x60 to 120x120 grids
- **Altar distribution**: 1-10 altars randomly placed
- **Open space navigation**: Requires exploration and path optimization

```python
# Example bucket configurations from the code:
maps = ["terrain_maps_nohearts"]
for size in ["large", "medium", "small"]:
    for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
        maps.append(f"varied_terrain/{terrain}_{size}")

dense_tasks.add_bucket("game.map_builder.instance_map.dir", maps)
dense_tasks.add_bucket("game.map_builder.instance_map.objects.altar", [Span(3, 50)])

sparse_tasks.add_bucket("game.map_builder.width", [Span(60, 120)])
sparse_tasks.add_bucket("game.map_builder.height", [Span(60, 120)])
sparse_tasks.add_bucket("game.map_builder.objects.altar", [Span(1, 10)])
```

### 3. **Training Dynamics**
- **Pathfinding optimization**: Agents learn efficient routes to goals
- **Exploration strategies**: Balancing exploration vs exploitation
- **Spatial reasoning**: Understanding map layouts and obstacles
- **Memory and planning**: Navigating complex mazes and remembering locations
- **Progressive difficulty**: From simple corridors to complex labyrinths

### 4. **Evaluation Suite**
The navigation curriculum includes a comprehensive evaluation suite with 23 different test environments:

#### ASCII Map Environments:
- **corridors**: Simple corridor navigation (450 steps max)
- **cylinder/cylinder_easy**: Circular navigation challenges
- **honeypot**: Attractive but potentially deceptive layouts
- **knotty**: Complex interconnected paths
- **memory_palace**: Requires spatial memory
- **obstacles series**: Progressive obstacle complexity (0-3)
- **radial series**: Radial maze patterns (large, mini, small, maze)
- **swirls**: Spiral navigation patterns
- **thecube**: 3D-like navigation in 2D space
- **walkaround**: Requires circumnavigation
- **wanderout**: Large open space navigation
- **labyrinth**: Complex maze navigation

#### Procedural Environments:
- **emptyspace_sparse**: Large open areas with sparse goals
- **emptyspace_outofsight**: Navigation beyond visual range
- **walls_outofsight/walls_withinsight**: Wall-based navigation with/without full visibility

### 5. **How to Use It**
You can train agents on the navigation curriculum using:

```bash
# Train on the navigation curriculum
./tools/run.py experiments.recipes.navigation.train --args run=my_experiment

# Evaluate trained agents on navigation suite
./tools/run.py experiments.recipes.navigation.eval --overrides policy_uris=wandb://run/my_experiment
```

## Purpose

The navigation curriculum serves as a **foundation for spatial intelligence** in the Metta AI framework. By mastering navigation skills, agents develop:

- **Efficient pathfinding algorithms**
- **Spatial memory and mapping**
- **Exploration strategies**
- **Goal-directed behavior**
- **Resource optimization**

These fundamental skills provide a base layer of intelligence that can be combined with social dynamics from the arena curriculum.

## Key Differences from Arena Curriculum

| Aspect              | Navigation Curriculum           | Arena Curriculum                    |
| ------------------- | ------------------------------- | ----------------------------------- |
| **Focus**           | Individual spatial intelligence | Multi-agent cooperation/competition |
| **Agents**          | 4 agents                        | 24 agents                           |
| **Environment**     | Navigation mazes and terrains   | Resource management and combat      |
| **Actions**         | Move, rotate, get_items         | Full action set including combat    |
| **Rewards**         | Heart collection from altars    | Complex resource conversion chains  |
| **Social Dynamics** | None                            | Kinship, alliances, combat          |
| **Difficulty**      | Spatial complexity              | Social complexity                   |

## Related Files in the Project

- `metta/experiments/recipes/navigation.py` - Main navigation curriculum implementation
- `metta/mettagrid/src/metta/mettagrid/config/envs.py` - Navigation environment configuration
- `metta/experiments/evals/navigation.py` - Navigation evaluation suite
- `metta/mettagrid/configs/maps/navigation/` - ASCII map files for evaluation
- `metta/metta/cogworks/curriculum/` - Curriculum learning system
- `metta/map/terrain_from_numpy.py` - Terrain generation utilities
