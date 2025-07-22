# Metta Configuration Guide

This guide explains the hierarchical configuration system used throughout Metta for experiments, environments, maps, curriculums, and training.

## Overview

Metta uses [Hydra](https://hydra.cc/) for configuration management, which provides:
- **Composition**: Combine multiple config files
- **Overrides**: Override any parameter from command line
- **Interpolation**: Reference other config values
- **Type Safety**: Structured configs with validation

## Configuration Hierarchy

```
configs/
├── agent/           # Agent architecture configs
├── env/             # Environment configurations
│   └── mettagrid/
│       ├── game/    # Game mechanics (objects, agents, etc.)
│       ├── curriculum/  # Learning curriculum definitions
│       └── map_builder/ # Map generation configs
├── trainer/         # Training hyperparameters
├── sim/             # Simulation/evaluation configs
├── wandb/           # Weights & Biases logging
├── sweep/           # Hyperparameter sweep configs
├── user/            # User-specific overrides
└── *.yaml           # Top-level job configs
```

## Key Configuration Types

### 1. Environment Configurations

Environment configs define the game world, rules, and objectives.

#### Basic Structure
```yaml
# configs/env/mettagrid/navigation/training/example.yaml
defaults:
  - /env/mettagrid/mettagrid@  # Base environment
  - _self_

sampling: 1  # Enable parameter sampling

game:
  num_agents: 4
  max_steps: 1000

  agent:
    view_dist: 7
    resource_limits:
      health: 100
    rewards:
      heart: 1.0
      ore_red: 0.1

  map_builder:
    _target_: metta.map.mapgen.MapGen
    width: 50
    height: 50
    root:
      type: metta.map.scenes.random.Random
      params:
        agents: 4
        objects:
          altar: 2
          wall: 20
```

#### Parameter Sampling
Use `${sampling:min,max,default}` for randomization:
```yaml
game:
  agent:
    rewards:
      ore_red: ${sampling:0.01,0.5,0.1}  # Samples between 0.01 and 0.5
```

### 2. Curriculum Configurations

Curriculums control task selection during training.

#### Random Curriculum
```yaml
_target_: metta.mettagrid.curriculum.random.RandomCurriculum

tasks:
  /env/mettagrid/navigation/easy: 1.0     # Weight 1.0
  /env/mettagrid/navigation/medium: 2.0   # Weight 2.0 (2x more likely)
  /env/mettagrid/navigation/hard: 1.0

env_overrides:
  game:
    num_agents: 16
```

#### Bucketed Curriculum
```yaml
_target_: metta.mettagrid.curriculum.bucketed.BucketedCurriculum

env_cfg_template: /env/mettagrid/navigation/base

buckets:
  # Parameter ranges
  game.map.width:
    range: [20, 100]
    bins: 4  # Creates: [20,40), [40,60), [60,80), [80,100]

  # Exact values
  game.agent.view_dist:
    values: [5, 7, 9]

  # Mixed types work too
  game.map_builder.room.objects.altar:
    range: [1, 10]
    bins: 3
```

#### Progressive Curriculum
```yaml
_target_: metta.mettagrid.curriculum.progressive.ProgressiveMultiTaskCurriculum

tasks:
  /env/easy: 1
  /env/medium: 1
  /env/hard: 1

performance_threshold: 0.8
progression_mode: "perf"  # "perf" or "time"
blending_smoothness: 0.5
```

### 3. Map Builder Configurations

Maps can be configured inline or loaded from external files.

#### Inline Map Configuration
```yaml
map_builder:
  _target_: metta.map.mapgen.MapGen
  width: 120
  height: 120

  root:
    type: metta.map.scenes.room_grid.RoomGrid
    params:
      rows: 3
      columns: 3
      border_width: 2

    children:
      - where:
          tags: ["room"]
        scene:
          type: metta.map.scenes.maze.Maze
          params:
            room_size: ["uniform", 1, 3]
```

#### External Scene Reference
```yaml
root:
  type: metta.map.scenes.random_scene.RandomScene
  params:
    candidates:
      - scene: /wfc/dungeons.yaml
        weight: 3
      - scene: /wfc/mazelike1.yaml
        weight: 1
```

### 4. Training Configurations

Training configs combine all components:

```yaml
# configs/train_job.yaml
defaults:
  - common
  - agent: fast
  - trainer: trainer
  - sim: arena
  - wandb: metta_research
  - _self_

trainer:
  curriculum: /env/mettagrid/curriculum/navigation/bucketed

  total_timesteps: 10_000_000_000
  batch_size: 524288
  minibatch_size: 16384

  checkpoint:
    checkpoint_interval: 50
    checkpoint_dir: ${run_dir}/checkpoints

  ppo:
    clip_coef: 0.1
    ent_coef: 0.01
    gamma: 0.99

run: my_experiment
seed: 42
```

### 5. User Configurations

User configs provide personal overrides:

```yaml
# configs/user/username.yaml
# @package __global__

defaults:
  - /common
  - /trainer/trainer
  - _self_

trainer:
  curriculum: /env/mettagrid/curriculum/navigation/progressive
  num_workers: 8
  total_timesteps: 1_000_000_000

wandb:
  entity: my-team
  project: my-project
  tags: ["navigation", "progressive"]

run: ${oc.env:USER}.${now:%Y%m%d_%H%M%S}
```

## Configuration Composition

### Defaults List

The `defaults` list controls config composition order:

```yaml
defaults:
  - base_config           # First: load base
  - override some/path    # Then: override specific path
  - optional: user        # Optional: may not exist
  - _self_               # Last: apply this file's values
```

### Path Notation

Use `@` to specify where configs are placed:

```yaml
defaults:
  - /env/mettagrid/mettagrid@game  # Load at 'game' key
  - agent/rewards@game.agent.rewards  # Nested placement
```

## Command Line Usage

### Basic Training

```bash
python train.py +user=myconfig
```

### Override Parameters

```bash
python train.py \
  trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed \
  trainer.total_timesteps=1000000 \
  game.num_agents=32
```

### Multiple Overrides

```bash
python train.py \
  +user=myconfig \
  trainer.batch_size=1048576 \
  'trainer.ppo.clip_coef=0.2' \
  'game.map_builder.root.params.num_agents=16'
```

### Sweep Configurations

```bash
python train.py --multirun \
  trainer.ppo.ent_coef=0.001,0.01,0.1 \
  trainer.ppo.clip_coef=0.1,0.2
```

## Best Practices

### 1. Organization

- **Group related configs** in subdirectories
- **Use descriptive names** that indicate purpose
- **Maintain consistent structure** across similar configs

### 2. Composition

- **Start with defaults** to avoid repetition
- **Override selectively** rather than redefining everything
- **Use `_self_`** to control override order

### 3. Parameter Naming

```yaml
# Good: Hierarchical and clear
game:
  agent:
    rewards:
      collectible: 0.1

# Avoid: Flat and ambiguous
agent_collectible_reward: 0.1
```

### 4. Documentation

```yaml
# Document complex parameters
game:
  agent:
    # View distance in grid cells
    # Higher values = more computational cost
    view_dist: 7  # default: 7, reasonable range: 5-11
```

### 5. Validation

```python
# Use structured configs for validation
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class GameConfig:
    num_agents: int = 1
    max_steps: int = 1000

cs = ConfigStore.instance()
cs.store(name="game_schema", node=GameConfig)
```

## Common Patterns

### Environment Variants

Create variants by extending base configs:

```yaml
# configs/env/mettagrid/navigation/easy.yaml
defaults:
  - base
  - _self_

game:
  max_steps: 500
  map_builder:
    root:
      params:
        objects:
          wall: 5  # Fewer obstacles
```

### Curriculum Composition

Combine multiple curriculum strategies:

```yaml
# First bucketize, then apply low-reward focus
_target_: metta.mettagrid.curriculum.low_reward.LowRewardCurriculum

tasks:
  /env/mettagrid/curriculum/navigation/bucketed_small: 1
  /env/mettagrid/curriculum/navigation/bucketed_large: 1
```

### Conditional Configuration

Use interpolation for conditional values:

```yaml
game:
  num_agents: 16
  map_builder:
    width: ${oc.select:game.large_map,200,50}
    height: ${oc.select:game.large_map,200,50}
```

## Debugging Configuration

### View Composed Config

```bash
python train.py --cfg job
```

### Save Composed Config

```bash
python train.py --cfg job > my_config.yaml
```

### Validate Configuration

```bash
python train.py --cfg job --package game.agent
```

## Related Documentation

- [Map Generation README](../metta/map/README.md) - Details on map configuration
- [Curriculum README](../metta/mettagrid/src/metta/mettagrid/curriculum/README.md) - Curriculum configuration
- [Hydra Documentation](https://hydra.cc/) - Official Hydra docs
