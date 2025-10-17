# Agora: Adaptive Curriculum Learning for RL

Agora provides a flexible, environment-agnostic curriculum learning system for reinforcement learning agents.

> **Name Origin**: Agora (ἀγορά) was the central public space in ancient Greek city-states where citizens gathered for assembly, learning, and the exchange of ideas. Similarly, this package serves as a central learning system where RL agents converge to learn and adapt through structured curriculum.

## Features

- 🎯 **Adaptive Task Generation**: Automatically generate tasks with varying difficulty
- 📊 **Learning Progress Tracking**: Monitor agent performance across curriculum
- 🔄 **Multiple Algorithms**: Built-in support for learning progress-based curriculum
- 🔌 **Framework Agnostic**: Works with any RL environment (Gymnasium, PettingZoo, etc.)
- ⚡ **Efficient**: Optimized for large-scale distributed training
- 🧩 **Modular**: Clean separation of concerns with extensible architecture

## Installation

```bash
# Basic installation
pip install agora

# With PufferLib support
pip install agora[puffer]

# With MettaGrid integration
pip install agora[mettagrid]

# All optional dependencies
pip install agora[all]

# For development
pip install agora[dev]
```

## Quick Start

```python
from agora import Curriculum, SingleTaskGenerator
from pydantic import BaseModel

# Define your task configuration (any pydantic model)
class MyTaskConfig(BaseModel):
    difficulty: int = 1
    num_obstacles: int = 5
    reward_scale: float = 1.0

# Create a simple curriculum
task_config = MyTaskConfig(difficulty=1)
generator = SingleTaskGenerator.Config(env=task_config)
curriculum = Curriculum(task_generator=generator)

# Sample tasks during training
for episode in range(1000):
    task = curriculum.sample_task()
    env.reset(config=task.config)
    # ... training loop ...
    curriculum.update_stats(task.id, episode_reward=reward)
```

## Learning Progress Curriculum

```python
from agora import (
    Curriculum,
    BucketedTaskGenerator,
    LearningProgressAlgorithm,
    LearningProgressConfig,
)

# Create bucketed task generator with increasing difficulty
base_config = MyTaskConfig(difficulty=1)
generator_config = BucketedTaskGenerator.Config(
    env=base_config,
    buckets={
        "difficulty": [1, 2, 3, 4, 5],
        "num_obstacles": [5, 10, 15, 20],
    },
    num_tasks_per_bucket=10
)

# Create curriculum with learning progress algorithm
lp_config = LearningProgressConfig(
    mode="bidirectional",  # Focus on tasks with highest learning progress
    capacity=100,
    min_episodes_per_task=5,
)

curriculum = Curriculum(
    task_generator=BucketedTaskGenerator(generator_config),
    algorithm=LearningProgressAlgorithm(lp_config)
)

# Training automatically adapts to agent's learning progress
for episode in range(10000):
    task = curriculum.sample_task()  # Selects based on learning progress
    # ... train ...
    curriculum.update_stats(task.id, episode_reward=reward)
```

## Core Concepts

### Task Generator
Task generators create task configurations with varying parameters:
- `SingleTaskGenerator`: Fixed task for baseline training
- `BucketedTaskGenerator`: Discrete difficulty levels
- `TaskGeneratorSet`: Combine multiple generators with weights

### Curriculum Algorithm
Algorithms determine task selection strategy:
- `LearningProgressAlgorithm`: Prioritizes tasks with highest learning progress
- Custom algorithms by extending `CurriculumAlgorithm`

### Tracking & Statistics
Built-in tracking for curriculum analysis:
- Per-task statistics (success rate, reward, etc.)
- Learning progress scoring
- Shared memory support for distributed training

## MettaGrid Integration

```python
from agora import Curriculum, BucketedTaskGenerator
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.envs import MettaGridEnv

# Create MettaGrid curriculum
mg_config = MettaGridConfig(
    # ... your config
)

generator_config = BucketedTaskGenerator.Config.from_mg(mg_config)
curriculum = Curriculum(task_generator=BucketedTaskGenerator(generator_config))

# Use with MettaGrid environment
task = curriculum.sample_task()
env = MettaGridEnv(env_cfg=task.config)
```

## PufferLib Integration

```python
from agora import Curriculum, CurriculumEnv

# Wrap with PufferEnv
curriculum_env = CurriculumEnv(
    curriculum=curriculum,
    # ... puffer kwargs
)

# Use in PufferLib training
```

## Documentation

- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Algorithms](docs/algorithms.md)
- [Examples](examples/)

## Contributing

Agora is part of the [Metta AI](https://github.com/Metta-AI/metta) project. Contributions are welcome!

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Agora in your research, please cite:

```bibtex
@software{agora2024,
  title={Agora: Adaptive Curriculum Learning for Reinforcement Learning},
  author={Metta AI Team},
  year={2024},
  url={https://github.com/Metta-AI/metta/tree/main/packages/agora}
}
```

