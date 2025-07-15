# Implementing Navigation Curriculum Systems: A Complete Guide

This guide provides step-by-step instructions for implementing structured navigation curriculums in Metta, based on lessons learned from creating center altar placement configurations and adaptive learning systems.

## Overview

Navigation curriculums enable agents to learn progressively from simple structured tasks to complex navigation challenges. This guide covers:

1. **Structured Task Creation**: Creating altar placement configurations with precise spatial control
2. **Curriculum Integration**: Combining different curriculum types for effective learning
3. **Room Sizing and Parameters**: Guidelines for optimal environment configuration
4. **Error Handling**: Common pitfalls and solutions
5. **Performance Optimization**: Best practices for efficient training

## System Architecture

```
┌─────────────────────┐
│ Training Pipeline   │
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │ Curriculum  │ ← Selects tasks (Random, Bucketed, Progressive, Learning Progress)
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │    Task     │ ← Contains environment config with map_builder
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ Environment │ ← Uses scene system with specific placement rules
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │     Map     │ ← Generated with center/random placement, proper spacing
    └─────────────┘
```

## Step 1: Create Structured Base Environments

### Center Altar Placement Configuration

Start with a base environment that uses precise altar placement:

```yaml
# configs/env/mettagrid/navigation/center_altar_base.yaml
defaults:
  - /env/mettagrid/mettagrid@
  - _self_

sampling: 1  # Enable parameter sampling for curriculum use

game:
  num_agents: 1  # Single agent for initial testing
  max_steps: 250

  agent:
    rewards:
      heart: 1.0

  map_builder:
    _target_: metta.map.mapgen.MapGen
    width: 60
    height: 60

    root:
      type: metta.map.scenes.layout.Layout
      params:
        # Create 3x3 grid of areas
        rows: 3
        columns: 3

        # Central 3x3 area for altar placement
        regions:
          middle:
            row_range: [1, 2]  # Middle row only
            col_range: [1, 2]  # Middle column only

        # Agent spawns in middle areas
        agents:
          middle: 1

        # Altars placed at exact center of middle areas
        objects:
          altar:
            regions: ["middle"]
            count: 3
            placement: center  # KEY: Precise center placement
```

**Key Design Principles:**
- **Precise Placement**: Use `placement: center` for structured, predictable altar locations
- **Spatial Separation**: Ensure agents can't see multiple altars simultaneously
- **Single Agent Start**: Begin with single agent to reduce complexity
- **Reward Focused**: Only reward hearts to focus learning on navigation

### Room Sizing Guidelines

Based on our testing, follow these room size requirements to ensure agents cannot see between rooms:

**Note**: Agent observation size is controlled by the environment configuration (`obs_width`/`obs_height`), not by individual agent parameters.

### Action Configuration

Actions should be configured as a dictionary with enabled flags:

```yaml
actions:
  attack:
    enabled: false
  swap:
    enabled: false
  change_color:
    enabled: false
  put_items:
    enabled: false
```

```yaml
# For different altar counts, use these minimum room sizes:

# 3 altars: 60x60 rooms (each grid cell ~18x18)
center_altar_3:
  map_builder:
    width: 60
    height: 60
  max_steps: 250

# 4 altars: 70x70 rooms (each grid cell ~21x21)
center_altar_4:
  map_builder:
    width: 70
    height: 70
  max_steps: 300

# 5 altars: 90x90 rooms (each grid cell ~27x27)
center_altar_5:
  map_builder:
    width: 90
    height: 90
  max_steps: 400

# 6 altars: 110x110 rooms (each grid cell ~33x33)
center_altar_6:
  map_builder:
    width: 110
    height: 110
  max_steps: 550

# 7 altars: 130x130 rooms (each grid cell ~39x39)
center_altar_7:
  map_builder:
    width: 130
    height: 130
  max_steps: 700

# 8 altars: 150x150 rooms (each grid cell ~45x45)
center_altar_8:
  map_builder:
    width: 150
    height: 150
  max_steps: 800
```


## Step 2: Create Individual Task Configurations

Create specific configurations for each altar count:

```yaml
# configs/env/mettagrid/navigation/center_altar_3.yaml
defaults:
  - center_altar_base
  - _self_

game:
  max_steps: 250
  map_builder:
    width: 60
    height: 60
    root:
      params:
        objects:
          altar:
            count: 3
```

```yaml
# configs/env/mettagrid/navigation/center_altar_4.yaml
defaults:
  - center_altar_base
  - _self_

game:
  max_steps: 300
  map_builder:
    width: 70
    height: 70
    root:
      params:
        objects:
          altar:
            count: 4
```

**Repeat for counts 5-8** with appropriate room sizes and step limits.

## Step 3: Create Curriculum Configurations

### Bucketed Curriculum for Exploration

Use bucketed curriculum to explore parameter combinations:

```yaml
# configs/env/mettagrid/curriculum/navigation/bucketed_center_altar.yaml
_target_: metta.mettagrid.curriculum.bucketed.BucketedCurriculum

env_cfg_template: /env/mettagrid/navigation/center_altar_base

buckets:
  # Map size variation
  game.map_builder.width:
    range: [60, 150]
    bins: 4

  game.map_builder.height:
    range: [60, 150]
    bins: 4

  # Altar count variation
  game.map_builder.root.params.objects.altar.count:
    range: [3, 8]
    bins: 6

  # Step limit variation
  game.max_steps:
    range: [250, 800]
    bins: 4

# Total combinations: 4 × 4 × 6 × 4 = 384 tasks
```

### Alternative: Single-Task Random Sampling Approach

As an alternative to bucketed curriculums, you can use a single configuration that randomly samples parameters from continuous ranges. Both approaches are valid:

```yaml
# configs/env/mettagrid/navigation/spiral_altar_random.yaml
defaults:
  - /env/mettagrid/navigation/evals/defaults@
  - _self_

sampling: 1  # Enable parameter sampling

game:
  num_agents: 1
  # Random max steps using sampling syntax: ${sampling:min, max, center}
  max_steps: ${sampling:400, 1000, 700}

  map_builder:
    _target_: metta.map.mapgen.MapGen
    # Random map size
    width: ${sampling:80, 120, 100}
    height: ${sampling:80, 120, 100}
    border_width: 1

    root:
      type: metta.map.scenes.spiral.Spiral
      params:
        objects:
          altar: ${sampling:4, 12, 8}  # Random 4-12 altars
        agents: 1
        spacing: ${sampling:12, 21, 16}  # Random spacing
        start_radius: 0
        radius_increment: ${sampling:2.0, 3.0, 2.5}  # Random tightness
        angle_increment: 0.3
        randomize_position: ${sampling:0, 4, 2}  # Random variation
        place_at_center: true
```

**When to use this approach:**
- **Simplicity**: When you want one configuration file instead of managing combinations
- **Continuous ranges**: When you need any value within the range, not just discrete bins
- **Exploration**: When you want maximum parameter diversity
- **Quick iteration**: When you're experimenting with parameter ranges

**When to use bucketed approach:**
- **Systematic coverage**: When you need to ensure specific parameter combinations are tested
- **Reproducibility**: When you want exact control over which values are used
- **Analysis**: When you need to track performance for specific parameter values
- **Curriculum progression**: When using progressive curriculums that need discrete steps

**Important: Parameter Sampling Syntax**
The correct syntax for random sampling is: `${sampling:min, max, center}`
- `min`: Minimum value
- `max`: Maximum value
- `center`: Center/default value used for scaling

### Single-Task Curriculum

Create a simple curriculum using the random task:

```yaml
# configs/env/mettagrid/curriculum/navigation/random_spiral_only.yaml
_target_: metta.mettagrid.curriculum.random.RandomCurriculum

tasks:
  # Single task with random parameters each episode
  /env/mettagrid/navigation/evals/spiral_altar_random: 1.0
```

### Learning Progress Curriculum for Adaptation

Create adaptive curriculum that focuses on challenging tasks:

```yaml
# configs/env/mettagrid/curriculum/navigation/learning_progress_center_altar.yaml
_target_: metta.mettagrid.curriculum.learning_progress.LearningProgressCurriculum

tasks:
  /env/mettagrid/navigation/center_altar_3: 1
  /env/mettagrid/navigation/center_altar_4: 1
  /env/mettagrid/navigation/center_altar_5: 1
  /env/mettagrid/navigation/center_altar_6: 1
  /env/mettagrid/navigation/center_altar_7: 1
  /env/mettagrid/navigation/center_altar_8: 1

# Focus on subset of tasks at once
num_active_tasks: 3
ema_timescale: 0.001
progress_smoothing: 0.05
rand_task_rate: 0.25
```

### Combined Curriculum for Comprehensive Training

Integrate structured tasks with standard navigation:

```yaml
# configs/env/mettagrid/curriculum/navigation/learning_progress_extended.yaml
_target_: metta.mettagrid.curriculum.learning_progress.LearningProgressCurriculum

tasks:
  # Original navigation tasks
  /env/mettagrid/navigation/training/terrain_from_numpy: 1
  /env/mettagrid/navigation/training/wfc_random: 1
  /env/mettagrid/navigation/training/maze_random: 1
  /env/mettagrid/navigation/training/rooms_random: 1
  /env/mettagrid/navigation/training/open_random: 1
  /env/mettagrid/navigation/training/varied_mix: 1

  # Center altar tasks for structured learning
  /env/mettagrid/navigation/center_altar_3: 1
  /env/mettagrid/navigation/center_altar_4: 1
  /env/mettagrid/navigation/center_altar_5: 1
  /env/mettagrid/navigation/center_altar_6: 1
  /env/mettagrid/navigation/center_altar_7: 1
  /env/mettagrid/navigation/center_altar_8: 1

# Focus on 5 tasks at once for balanced learning
num_active_tasks: 5
ema_timescale: 0.001
progress_smoothing: 0.05
rand_task_rate: 0.25
```

## Step 4: Multi-Agent Configuration

Scale to multi-agent environments for production training:

```yaml
# Multiroom configuration using MultiRoom class
game:
  num_agents: 4
  max_steps: ${sampling:400, 1000, 700}

  # Actions should be configured with enabled flags
  actions:
    attack:
      enabled: false
    swap:
      enabled: false
    change_color:
      enabled: false
    put_items:
      enabled: false

  agent:
    rewards:
      heart: 1.0
    # Do NOT add view_shape or clip_shape here

  map_builder:
    _target_: metta.mettagrid.room.multi_room.MultiRoom
    num_rooms: ${..num_agents}
    border_width: 8
    room:
      _target_: metta.map.mapgen.MapGen
      width: ${sampling:50, 70, 60}
      height: ${sampling:50, 70, 60}
      border_width: 4

      root:
        type: metta.map.scenes.your_scene.YourScene
        params:
          objects:
            altar: ${sampling:5, 25, 12}
          agents: 1  # One agent per room
```

## Step 5: Common Issues and Solutions

### Issue 1: "CurriculBucketedum" Class Not Found

**Problem**: Typo in curriculum class name
**Solution**: Use exact class name from implementation

```yaml
# WRONG
_target_: metta.mettagrid.curriculum.bucketed.CurriculBucketedum

# CORRECT
_target_: metta.mettagrid.curriculum.bucketed.BucketedCurriculum
```

### Issue 2: "0 agents in map" Error

**Problem**: Agent placement not configured properly
**Solution**: Add explicit agent placement in scene configuration

```yaml
# Add to scene params:
agents:
  middle: 1  # Spawn 1 agent in middle regions

# Or for random placement:
agents: 1  # Spawn 1 agent randomly
```

### Issue 3: No Episode Recordings

**Problem**: Missing sampling parameter
**Solution**: Add sampling to environment config

```yaml
# Add to environment config:
sampling: 1  # Enable episode recording
```

### Issue 4: Validation Errors in ChildrenAction

**Problem**: Missing scene field in nested actions
**Solution**: Add scene field to all ChildrenAction configurations

```yaml
# In Layout scene children:
children:
  - where:
      tags: ["room"]
    scene:  # ADD THIS FIELD
      type: metta.map.scenes.random.Random
      params:
        objects:
          altar: 1
```

## Step 6: Testing and Validation

### Test Single Environment
```bash
# Test a single environment configuration
python -m tools.train \
  run=test_center_altar \
  trainer.curriculum=/env/mettagrid/navigation/center_altar_3 \
  trainer.total_timesteps=1000 \
  trainer.num_workers=1 \
  wandb=off
```

### Test Curriculum
```bash
# Test curriculum with minimal training
python -m tools.train \
  run=test_curriculum \
  trainer.curriculum=/env/mettagrid/curriculum/navigation/learning_progress_center_altar \
  trainer.total_timesteps=10000 \
  trainer.num_workers=1 \
  wandb=off
```

### Validate Configuration
```bash
# Check composed configuration
python train.py --cfg job \
  trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed_center_altar
```

### Preview Maps
```bash
# Generate and view maps
python -m tools.map.gen configs/env/mettagrid/navigation/center_altar_3.yaml
```

## Step 7: Performance Optimization

### Batch Size Considerations
```yaml
# For multi-agent environments, ensure batch size accommodates all agents
trainer:
  batch_size: 524288  # Large enough for parallel agents
  minibatch_size: 16384
  num_workers: 32
```

### Memory Management
```yaml
# For Learning Progress curriculum, limit memory usage
curriculum:
  memory: 25  # Keep last 25 episode scores
  sample_threshold: 10  # Minimum samples before task selection
```

### Task Selection Efficiency
```yaml
# Use focused task selection for faster convergence
curriculum:
  num_active_tasks: 3  # Focus on 3 tasks at once
  rand_task_rate: 0.25  # 25% random exploration
```

## Step 8: Integration Patterns

### Pattern 1: Structured → Random Transition
```yaml
# Start with structured tasks, then transition to random
_target_: metta.mettagrid.curriculum.progressive.ProgressiveMultiTaskCurriculum

tasks:
  /env/mettagrid/navigation/center_altar_3: 1
  /env/mettagrid/navigation/center_altar_4: 1
  /env/mettagrid/navigation/center_altar_5: 1
  /env/mettagrid/navigation/training/varied_mix: 1

performance_threshold: 0.8
progression_mode: "perf"
```

### Pattern 2: Multi-Level Buckets
```yaml
# Create buckets within buckets for fine-grained control
buckets:
  # Coarse-grained map size
  game.map_builder.width:
    range: [60, 150]
    bins: 3

  # Fine-grained altar count within each size
  game.map_builder.root.params.objects.altar.count:
    range: [3, 8]
    bins: 6

  # Correlated step limits
  game.max_steps:
    values: [250, 300, 400, 550, 700, 800]
```

### Pattern 3: Combining Buckets with Random Sampling
```yaml
# Use buckets for some parameters and random sampling for others
_target_: metta.mettagrid.curriculum.bucketed.BucketedCurriculum

env_cfg_template: /env/mettagrid/navigation/spiral_altar_base

buckets:
  # Use discrete buckets for map sizes
  game.map_builder.width:
    values: [80, 100, 120]

  game.map_builder.height:
    values: [80, 100, 120]

# Use random sampling within each bucket task
env_overrides:
  game:
    map_builder:
      root:
        params:
          objects:
            altar: ${sampling:4, 12, 8}  # Random within each bucket
          spacing: ${sampling:12, 21, 16}
          radius_increment: ${sampling:2.0, 3.0, 2.5}
```

### Pattern 4: Adaptive Focus
```yaml
# Combine low-reward focus with learning progress
_target_: metta.mettagrid.curriculum.low_reward.LowRewardCurriculum

tasks:
  /env/mettagrid/curriculum/navigation/bucketed_center_altar: 1

# Automatically focus on challenging parameter combinations
moving_avg_decay_rate: 0.01
```

## Step 9: Monitoring and Debugging

### Essential Logging
```python
# Monitor curriculum statistics
stats = curriculum.get_curriculum_stats()
# Key metrics: task_weights, smoothed_performance, progress_rates
```

### Debug Checklist
1. **✓ Class names exact** - Check for typos in `_target_` fields
2. **✓ Agent placement** - Verify agents spawn in maps
3. **✓ Room sizing** - Ensure view distance constraints met
4. **✓ Sampling enabled** - Add `sampling: 1` for recordings
5. **✓ Scene fields** - Add `scene:` to all ChildrenAction configs
6. **✓ Step limits** - Match step limits to map complexity
7. **✓ Resource limits** - Ensure agents can carry collected items

### Performance Monitoring
```bash
# Track training progress
tail -f logs/training.log | grep -E "(reward|curriculum|task)"

# Monitor curriculum task selection
wandb login
# Check "curriculum/task_weights" and "curriculum/smoothed_performance"
```

## Step 10: Advanced Techniques

### Dynamic Room Sizing
```yaml
# Use interpolation for dynamic parameter adjustment
game:
  map_builder:
    width: ${eval:"60 + 10 * (${game.map_builder.root.params.objects.altar.count} - 3)"}
    height: ${eval:"60 + 10 * (${game.map_builder.root.params.objects.altar.count} - 3)"}
```

### Conditional Curriculum Selection
```yaml
# Switch curriculum based on performance
_target_: metta.mettagrid.curriculum.conditional.ConditionalCurriculum

curricula:
  warmup:
    curriculum: /env/mettagrid/curriculum/navigation/learning_progress_center_altar
    condition: "steps < 1000000"

  main:
    curriculum: /env/mettagrid/curriculum/navigation/learning_progress_extended
    condition: "steps >= 1000000"
```

### Hierarchical Task Decomposition
```yaml
# Break complex tasks into subtasks
tasks:
  # Phase 1: Single altar
  /env/mettagrid/navigation/center_altar_1: 1

  # Phase 2: Multiple altars, small maps
  /env/mettagrid/navigation/center_altar_3: 1
  /env/mettagrid/navigation/center_altar_4: 1

  # Phase 3: Multiple altars, large maps
  /env/mettagrid/navigation/center_altar_6: 1
  /env/mettagrid/navigation/center_altar_8: 1
```

## Troubleshooting Guide

### Common Error Messages and Solutions

**"No module named 'metta.mettagrid.curriculum.bucketed'"**
- Solution: Check import path and ensure module exists
- Verify: `ls mettagrid/src/metta/mettagrid/curriculum/bucketed.py`

**"agents must be >= 1"**
- Solution: Add agent placement to scene configuration
- Add: `agents: 1` or `agents: {middle: 1}` to scene params

**"Extra inputs are not permitted"** (e.g., for `agent.view_dist`)
- Solution: Remove invalid configuration fields
- Note: `view_dist` is not a valid agent parameter
- Do not add `view_shape` or `clip_shape` to agent config
- Check the schema for valid fields under each configuration section

**"Expected DictConfig, got str"**
- Solution: Check YAML indentation and structure
- Verify: All nested configs properly formatted

**"Task selection probability is 0"**
- Solution: Check task weights and ensure tasks are valid
- Verify: All task paths exist and are loadable

**"Map generation failed"**
- Solution: Check room sizing and object placement constraints
- Verify: Room size allows for proper object placement

## Scene Implementation Patterns

When creating custom scenes, follow these patterns:

```python
from metta.config import Config
from metta.map.scene import Scene

class YourSceneParams(Config):
    """Parameters for your scene."""
    objects: dict[str, int] = {"altar": 10}  # Use dict for objects
    agents: int | dict[str, int] = 0
    your_param: int = 5

class YourScene(Scene[YourSceneParams]):
    """Your scene description."""

    def render(self):
        """Use render() method, not create_grid()."""
        height, width, params = self.height, self.width, self.params

        # Collect objects
        symbols = []
        for obj_name, count in params.objects.items():
            symbols.extend([obj_name] * count)

        # Handle agents
        if isinstance(params.agents, int):
            agents = ["agent.agent"] * params.agents
        elif isinstance(params.agents, dict):
            agents = ["agent." + str(agent) for agent, na in params.agents.items() for _ in range(na)]

        # Place items on self.grid[y, x] = symbol
        # Check if self.grid[y, x] == "empty" before placing
```

## Best Practices Summary

1. **Start Simple**: Begin with single-agent, small maps
2. **Test Incrementally**: Validate each component before combining
3. **Size Appropriately**: Follow room sizing guidelines for view distance
4. **Monitor Performance**: Track curriculum statistics and task selection
5. **Use Structured Progression**: Combine center altar tasks with random environments
6. **Validate Configurations**: Test configs before long training runs
7. **Document Parameters**: Keep notes on effective parameter combinations
8. **Version Control**: Track curriculum configs alongside code changes
9. **Actions Format**: Use dict with enabled flags, not lists of action names
10. **Multiroom Setup**: Use MultiRoom class for multi-agent environments
11. **Scene Parameters**: Use objects dict and agents field in scene params

## Related Documentation

- [Curriculum System README](../mettagrid/src/metta/mettagrid/curriculum/README.md) - Complete curriculum implementation details
- [Configuration Guide](../configs/README.md) - Hydra configuration system
- [Map Generation README](../metta/map/README.md) - Scene system and map building

## Quick Reference Commands

```bash
# Create new curriculum
python train.py trainer.curriculum=/path/to/curriculum

# Test configuration
python train.py --cfg job trainer.curriculum=/path/to/curriculum

# Override parameters
python train.py trainer.curriculum=/path/to/curriculum \
  game.num_agents=8 \
  game.max_steps=1000

# Generate map preview
python -m tools.map.gen configs/env/path/to/config.yaml

# Validate environment
python -m tools.train run=test trainer.curriculum=/path/to/curriculum \
  trainer.total_timesteps=100 wandb=off
```

This guide provides a complete workflow for implementing navigation curriculum systems based on practical experience with center altar placement configurations and adaptive learning strategies.
