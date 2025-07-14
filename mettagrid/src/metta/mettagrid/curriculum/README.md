# Curriculum Learning System

The Metta curriculum learning system provides adaptive task selection strategies for multi-agent reinforcement learning, enabling agents to learn progressively through carefully structured task sequences.

## Overview

Curriculum learning helps agents learn complex behaviors by presenting tasks in a structured sequence, from simple to complex. The system supports multiple scheduling strategies, from fixed distributions to adaptive algorithms that respond to agent performance.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Curriculum Type │────▶│   Task Selection │────▶│ Environment Task │
│   (Strategy)    │     │   (get_task())   │     │  (with config)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         ▲                                               │
         │                                               │
         └───────────────────────────────────────────────┘
                    Performance Feedback
                    (complete_task())
```

## Core Components

### 1. Base Classes

#### Curriculum (Abstract Base)
```python
class Curriculum:
    def get_task(self) -> Task:
        """Select next task for training"""

    def complete_task(self, id: str, score: float):
        """Update curriculum based on task performance"""

    def get_curriculum_stats(self) -> dict:
        """Return statistics for logging"""
```

#### Task
Encapsulates an environment configuration with metadata:
```python
class Task:
    def __init__(self, id: str, curriculum: Curriculum, env_cfg: DictConfig):
        self._id = id
        self._curriculum = curriculum
        self._env_cfg = env_cfg
```

### 2. Curriculum Types

#### RandomCurriculum
Fixed-weight sampling from multiple tasks:
```yaml
_target_: metta.mettagrid.curriculum.random.RandomCurriculum
tasks:
  /env/mettagrid/navigation/easy: 1.0
  /env/mettagrid/navigation/medium: 2.0
  /env/mettagrid/navigation/hard: 1.0
```

#### BucketedCurriculum
Generates task variations through parameter sweeps:
```yaml
_target_: metta.mettagrid.curriculum.bucketed.BucketedCurriculum
env_cfg_template: /env/mettagrid/navigation/base

buckets:
  game.map.width:
    range: [20, 100]
    bins: 4  # Creates 4 buckets: [20,40), [40,60), [60,80), [80,100]
  game.map.height:
    range: [20, 100]
    bins: 4
  game.max_steps:
    values: [500, 1000, 1500]  # Exact values
```

#### ProgressiveMultiTaskCurriculum
Sequential task progression based on performance:
```yaml
_target_: metta.mettagrid.curriculum.progressive.ProgressiveMultiTaskCurriculum
tasks:
  /env/easy: 1
  /env/medium: 1
  /env/hard: 1

# Progression parameters
performance_threshold: 0.8
progression_mode: "perf"  # "perf" or "time"
blending_mode: "logistic"  # "logistic" or "linear"
smoothing: 0.1
progression_rate: 0.001
```

#### LowRewardCurriculum
Focuses on challenging tasks with low average rewards:
```yaml
_target_: metta.mettagrid.curriculum.low_reward.LowRewardCurriculum
tasks:
  /env/task1: 1
  /env/task2: 1

moving_avg_decay_rate: 0.01  # Exponential moving average
```

#### LearningProgressCurriculum
Adaptive sampling based on learning progress (rate of improvement):
```yaml
_target_: metta.mettagrid.curriculum.learning_progress.LearningProgressCurriculum
tasks:
  /env/task1: 1
  /env/task2: 1

# Learning progress parameters
ema_timescale: 0.001
progress_smoothing: 0.05
num_active_tasks: 16
rand_task_rate: 0.25
sample_threshold: 10
memory: 25
```

## How Curriculums Work

### 1. Task Selection Process

```python
# In training loop
task = curriculum.get_task()
env_cfg = task.env_cfg()
env = create_environment(env_cfg)

# Train on environment...
episode_reward = train_episode(env)

# Provide feedback
curriculum.complete_task(task.id(), episode_reward)
```

### 2. Bucketed Curriculum Generation

The `BucketedCurriculum` automatically generates task variations:

```python
# For buckets with ranges:
buckets = {
    "param": {"range": [10, 50], "bins": 2}
}
# Generates: [(10, 30), (30, 50)]

# For buckets with values:
buckets = {
    "param": {"values": [1, 2, 3]}
}
# Uses exact values: [1, 2, 3]

# Cartesian product creates all combinations
```

### 3. Automated Scheduling

Different curriculums implement different scheduling strategies:

#### Performance-Based (Progressive)
- Tracks smoothed performance across tasks
- Advances when performance exceeds threshold
- Uses blending functions for smooth transitions

#### Learning Progress
- Uses fast/slow exponential moving averages
- Measures rate of improvement
- Focuses on tasks with high learning potential

#### Low Reward Focus
- Maintains reward statistics per task
- Weights tasks by `max_reward / average_reward`
- Automatically focuses on challenging tasks

## Configuration Examples

### Navigation Curriculum with Buckets
```yaml
_target_: metta.mettagrid.curriculum.bucketed.BucketedCurriculum
env_cfg_template: /env/mettagrid/navigation/training/terrain_from_numpy

buckets:
  # Vary map types
  game.map_builder.room.dir:
    values: [
      "terrain_maps_nohearts",
      "varied_terrain/balanced_large",
      "varied_terrain/maze_small",
      "varied_terrain/dense_medium"
    ]

  # Vary difficulty parameters
  game.map_builder.room.objects.altar:
    range: [10, 50]
    bins: 3

  game.agent.view_dist:
    values: [5, 7, 9]

env_overrides:
  game:
    num_agents: 16
```

### Progressive Object Use Curriculum
```yaml
_target_: metta.mettagrid.curriculum.progressive.ProgressiveMultiTaskCurriculum

tasks:
  /env/mettagrid/object_use/collect_only: 1
  /env/mettagrid/object_use/simple_craft: 1
  /env/mettagrid/object_use/complex_craft: 1
  /env/mettagrid/object_use/pvp_arena: 1

performance_threshold: 0.7
progression_mode: "perf"
blending_smoothness: 0.3
```

### Adaptive Learning Progress
```yaml
_target_: metta.mettagrid.curriculum.learning_progress.LearningProgressCurriculum

tasks:
  /env/mettagrid/navigation/terrain1: 1
  /env/mettagrid/navigation/terrain2: 1
  /env/mettagrid/navigation/terrain3: 1
  /env/mettagrid/navigation/terrain4: 1

num_active_tasks: 8  # Focus on subset of tasks
rand_task_rate: 0.1  # 10% random exploration
```

## Integration with Training

### Trainer Configuration
```yaml
trainer:
  curriculum: /env/mettagrid/curriculum/navigation/bucketed

  # Other trainer settings...
  total_timesteps: 10_000_000_000
  num_workers: 32
```

### Using Curriculum in Code
```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Load curriculum
cfg = OmegaConf.load("curriculum_config.yaml")
curriculum = instantiate(cfg)

# Training loop
for epoch in range(num_epochs):
    # Get task from curriculum
    task = curriculum.get_task()

    # Create environment with task config
    env = create_env(task.env_cfg())

    # Train and get score
    score = train_on_env(env)

    # Update curriculum
    curriculum.complete_task(task.id(), score)

    # Log statistics
    stats = curriculum.get_curriculum_stats()
    log_metrics(stats)
```

## Creating Custom Curriculums

### 1. Extend Base Class
```python
from metta.mettagrid.curriculum.core import Curriculum, Task

class MyCustomCurriculum(Curriculum):
    def __init__(self, config):
        self._config = config
        self._task_stats = {}

    def get_task(self) -> Task:
        # Implement custom task selection logic
        task_id = self._select_task()
        task_cfg = self._get_task_config(task_id)
        return Task(task_id, self, task_cfg)

    def complete_task(self, id: str, score: float):
        # Update internal statistics
        self._task_stats[id] = score
        self._update_selection_weights()
```

### 2. Add Configuration
```yaml
_target_: my_module.MyCustomCurriculum
my_parameter: value
task_configs:
  - path: /env/task1
    weight: 1.0
```

## Best Practices

1. **Start Simple**: Begin with `RandomCurriculum` to establish baselines
2. **Use Buckets for Exploration**: `BucketedCurriculum` helps discover effective parameter ranges
3. **Progress Gradually**: Use `ProgressiveCurriculum` for known skill progressions
4. **Adapt to Performance**: Use `LearningProgressCurriculum` for automatic adaptation
5. **Monitor Statistics**: Track curriculum stats to understand task selection
6. **Version Control Configs**: Keep curriculum configs in version control for reproducibility

## Monitoring and Debugging

### Curriculum Statistics
Most curriculums provide statistics via `get_curriculum_stats()`:

- **Task probabilities**: Current selection weights
- **Completion rates**: How often each task is selected
- **Performance metrics**: Task-specific scores
- **Learning progress**: Rate of improvement

### Logging Integration
```python
# Curriculum stats are automatically logged in training
stats = curriculum.get_curriculum_stats()
# Logs: smoothed_performance, progress, task_weights, etc.
```

## Performance Tips

1. **Batch Size Considerations**: Ensure batch size accommodates all agents across parallel environments
2. **Task Diversity**: Balance exploration vs exploitation in task selection
3. **Smooth Transitions**: Use appropriate smoothing parameters to avoid abrupt changes
4. **Memory Management**: For `LearningProgressCurriculum`, adjust memory parameter based on training frequency

## Relationship with Maps

Curriculums often vary map configurations as part of task difficulty. See the [Map Generation README](../../../../../map/README.md) for details on map parameters that can be varied in curriculums.

## Reference: Entity Representations and Parameter Paths

### Agent Representations

Agents are represented differently in configuration vs grid:

**In Configuration:**
```yaml
# As integer (creates default agents)
agents: 4  # Creates 4 "agent.agent" entities

# As dictionary (creates specific agent types)
agents:
  agent: 2      # 2 default agents
  team_1: 3     # 3 team_1 agents
  team_2: 3     # 3 team_2 agents
```

**In Grid** (from `char_encoder.py`):
- `"agent.agent"` - Default agent (renders as `@` or `0-9`)
- `"agent.team_1"` - Team 1 agent (renders as `1`)
- `"agent.team_2"` - Team 2 agent (renders as `2`)
- `"agent.prey"` - Prey agent (renders as `p`)
- `"agent.predator"` - Predator agent (renders as `P`)

**Character Encoding** (from `nethack.py` renderer):
- Agents 0-9: Rendered as their ID number
- Agents 10+: Rendered as letters (a-z)

### Valid Object Types

**Basic Objects** (from `configs/env/mettagrid/game/objects/`):
- `wall` - Impassable barrier (type_id: 1)
- `block` - Movable/swappable block (type_id: 14)
- `altar` - Converts resources to hearts (type_id: 8)

**Resource Objects**:
- `mine_red` - Produces ore_red (type_id: 2)
- `mine_blue` - Produces ore_blue (type_id: 3)
- `mine_green` - Produces ore_green (type_id: 4)
- `generator_red` - Converts ore to battery (type_id: 5)
- `generator_blue` - Converts ore to battery (type_id: 6)
- `generator_green` - Converts ore to battery (type_id: 7)

**Combat Objects**:
- `armory` - Produces armor (type_id: 9)
- `lasery` - Produces laser weapons (type_id: 10)

**Advanced Objects**:
- `lab` - Produces blueprints (type_id: 11)
- `factory` - Mass production (type_id: 12)
- `temple` - Advanced conversions (type_id: 13)

### Distribution Types

**Float Distributions** (from `map/random/float.py`):
```yaml
# Constant value
parameter: 5.0

# Uniform distribution
parameter: ["uniform", 0.1, 0.5]  # Between 0.1 and 0.5

# Log-normal distribution (90% between low and high)
parameter: ["lognormal", 0.001, 0.01]  # 90% between 0.001 and 0.01
parameter: ["lognormal", 0.001, 0.01, 0.05]  # With max cap at 0.05
```

**Integer Distributions** (from `map/random/int.py`):
```yaml
# Constant value
parameter: 10

# Uniform distribution (inclusive)
parameter: ["uniform", 5, 15]  # Between 5 and 15 inclusive
```

### Object Placement Rules

**RandomObjects Scene** (from `scenes/random_objects.py`):
- Uses **percentages/density** of total map area
- Example: `mine_red: 0.01` means 1% of map cells

**Random Scene** (from `scenes/random.py`):
- Uses **exact counts**
- Example: `wall: 20` means exactly 20 walls
- Automatically reduces counts if they exceed 2/3 of map area

### Valid Parameter Paths

Based on `mettagrid_config.py`, here are the valid game configuration paths:

**Game Parameters**:
```yaml
game.num_agents: int (≥1)
game.max_steps: int (≥0, 0=unlimited)
game.obs_width: int (≥1)
game.obs_height: int (≥1)
```

**Agent Parameters**:
```yaml
game.agent.default_resource_limit: int (≥0)
game.agent.freeze_duration: int (≥-1)
game.agent.action_failure_penalty: float (≥0)
game.agent.resource_limits.<resource>: int
game.agent.rewards.<resource>: float
game.agent.rewards.<resource>_max: int
```

**Map Builder Parameters**:
```yaml
game.map_builder._target_: str  # Class name
game.map_builder.width: int
game.map_builder.height: int
game.map_builder.border_width: int
game.map_builder.root: dict  # Scene configuration
game.map_builder.room.*  # Room-specific parameters
```

**Object Parameters** (when using scene parameters):
```yaml
game.map_builder.root.params.objects.<object_name>: int/float
game.map_builder.room.objects.<object_name>: int/float
```

### Testing and Validation

**Configuration Validation**:
```bash
# Validate a configuration file
python tools/validate_config.py env/mettagrid/navigation/easy

# Test a specific bucket configuration
python train.py --cfg job \
  trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed
```

**Map Preview**:
```bash
# Generate and view a map from config
python -m tools.map.gen configs/env/mettagrid/map_builder/auto.yaml

# View specific bucket's map
python -m tools.map.gen <config_path> \
  game.map_builder.width=50 \
  game.map_builder.height=50
```

**Environment Testing**:
```bash
# Quick environment test with minimal training
python -m tools.train \
  run=test_curriculum \
  trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed \
  trainer.total_timesteps=100 \
  trainer.num_workers=1 \
  wandb=off
```

### Common Parameter Constraints

1. **Agent Density**: Generally keep below 10% of map area
   - `num_agents <= map_width * map_height * 0.1`

2. **Object Counts**: Random scene reduces if > 66% of map
   - Total objects should be < `map_area * 0.66`

3. **View Distance**: Limited by observation window
   - `agent.view_dist` typically 5-11
   - Must fit in `obs_width` and `obs_height`

4. **Resource Limits**: Must include all inventory items
   - Default: `agent.default_resource_limit`
   - Override: `agent.resource_limits.<item>`

5. **Map Dimensions**: Consider performance
   - Small: 20-40
   - Medium: 40-80
   - Large: 80-200

### Bucket Generation Example

Complete example showing all concepts:

```yaml
_target_: metta.mettagrid.curriculum.bucketed.BucketedCurriculum
env_cfg_template: /env/mettagrid/navigation/base

buckets:
  # Map size variation
  game.map_builder.width:
    range: [30, 90]
    bins: 3  # Creates: [30,50), [50,70), [70,90]

  # Agent configuration
  game.num_agents:
    values: [4, 8, 16]

  # Use different agent types
  game.map_builder.room.agents:
    values:
      - 4  # 4 default agents
      - {agent: 2, team_1: 2}  # Mixed teams
      - {team_1: 2, team_2: 2}  # Versus mode

  # Object density (for RandomObjects scene)
  game.map_builder.root.params.object_ranges.wall:
    values:
      - ["uniform", 0.05, 0.1]  # 5-10% walls
      - ["uniform", 0.1, 0.2]   # 10-20% walls

  # Object count (for Random scene)
  game.map_builder.room.objects.altar:
    range: [1, 5]
    bins: 2  # [1,3) and [3,5]

  # Agent parameters
  game.agent.view_dist:
    values: [5, 7, 9]

  game.agent.rewards.ore_red:
    values: [0.01, 0.05, 0.1]

# Override to ensure consistency
env_overrides:
  game:
    obs_width: 11  # Must accommodate largest view_dist
    obs_height: 11
```

This will generate 3×3×3×2×2×3×3 = 972 different task variations!
