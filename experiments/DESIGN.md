# Experiments Framework

A Python-based framework for defining, launching, managing, and analyzing distributed training experiments on Skypilot.

## Overview

The experiments framework provides:

- **Type-safe configuration** using Pydantic models that serialize to Hydra-compatible YAML
- **Clean separation** between infrastructure (Skypilot) and training (Hydra) configuration
- **YAML-based config transfer** via Skypilot's file_mounts (no git commits required)
- **Service layer architecture** for integration with analysis tools

## Quick Start

### Running an Existing Experiment

```bash
# Preview what will be launched
uv run experiments/recipes/arena_experiment.py --no-launch

# Launch to Skypilot
uv run experiments/recipes/arena_experiment.py

# Customize infrastructure
uv run experiments/recipes/arena_experiment.py --gpus=4 --nodes=2 --spot=false
```

### Creating a Simple A/B Test

Create a new experiment that launches multiple jobs with different parameters:

```python
# experiments/recipes/lr_ab_test.py
from typing import List
from experiments.experiment import Experiment, ExperimentConfig
from experiments.training_job import TrainingJobConfig
from experiments.training_run_config import TrainingRunConfig
from experiments.skypilot_job_config import SkypilotJobConfig
from metta.rl.trainer_config import TrainerConfig, OptimizerConfig, TorchProfilerConfig
from metta.sim.simulation_config import SimulationConfig

class LearningRateABTest(Experiment):
    """A/B test comparing two learning rates."""

    def training_job_configs(self) -> List[TrainingJobConfig]:
        configs = []
        for lr in [0.0001, 0.0005]:  # Test two learning rates
            trainer = TrainerConfig(
                num_workers=4,
                optimizer=OptimizerConfig(learning_rate=lr),
                simulation=SimulationConfig(replay_dir="${run_dir}/replays"),
                profiler=TorchProfilerConfig(profile_dir="${run_dir}/torch_traces"),
            )
            training = TrainingRunConfig(
                curriculum="env/mettagrid/curriculum/arena/basic",
                trainer=trainer,
                wandb_tags=["ab_test", f"lr_{lr}"],
            )
            configs.append(TrainingJobConfig(
                skypilot=SkypilotJobConfig(
                    gpus=self.config.gpus,
                    nodes=self.config.nodes,
                    spot=self.config.spot,
                ),
                training=training,
            ))
        return configs

class LearningRateABTestConfig(ExperimentConfig):
    name: str = "lr_ab_test"
    # Infrastructure params passed by runner
    gpus: int = 1
    nodes: int = 1
    spot: bool = True
    git_check: bool = True

if __name__ == "__main__":
    from experiments.runner import runner
    runner(LearningRateABTest, LearningRateABTestConfig)
```

## Architecture

### Configuration Hierarchy

```
ExperimentConfig                    # Base: name, launch, instance_name
├── SingleJobExperimentConfig       # Inherits from both ExperimentConfig and TrainingJobConfig
│   └── ArenaExperimentConfig       # Specific experiment implementation
└── LearningRateABTestConfig        # Multi-job experiment config

TrainingJobConfig                   # Complete job specification
├── SkypilotJobConfig              # Infrastructure: GPUs, nodes, spot instances
└── TrainingRunConfig              # Training: agent, curriculum, hyperparameters
    └── TrainerConfig              # Nested: optimizer, PPO, checkpointing
```

### How Config Transfer Works

1. **Local**: Python config objects are created and validated
2. **Serialization**: `TrainingRunConfig.serialize_to_yaml_file()` creates a YAML in `configs/experiments/`
3. **Transfer**: Skypilot mounts the YAML and copies it to `configs/experiments/` on remote
4. **Execution**: Remote training loads via `+experiments={instance_name}` Hydra override

Example serialized YAML structure:

```yaml
defaults:
  - ../common # Relative paths from experiments/
  - ../agent/fast
  - ../trainer/trainer
  - ../sim/arena
  - ../wandb/metta_research
  - _self_

trainer:
  curriculum: env/mettagrid/curriculum/arena/basic
  total_timesteps: 10000000
  # ... full trainer config
```

## Advanced Usage

### Custom Training Configuration

For experiments requiring specific hyperparameters:

```python
from metta.rl.trainer_config import (
    TrainerConfig, OptimizerConfig, TorchProfilerConfig
from metta.sim.simulation_config import SimulationConfig

trainer = TrainerConfig(
    total_timesteps=50_000_000,
    batch_size=8192,
    minibatch_size=256,
    num_workers=8,
    optimizer=OptimizerConfig(
        type="muon",
        learning_rate=0.001,
    ),
    simulation=SimulationConfig(replay_dir="${run_dir}/replays"),
    profiler=TorchProfilerConfig(profile_dir="${run_dir}/torch_traces"),
)

training = TrainingRunConfig(
    curriculum="env/mettagrid/curriculum/arena/basic",
    trainer=trainer,
)
```

## Service Layer

The framework provides service classes for programmatic access:

```python
from experiments.skypilot_service import get_skypilot_service
from experiments.wandb_service import get_wandb_service

# Query job status
sky_service = get_skypilot_service()
job_status = sky_service.get_job_status("sky-job-id")
wandb_run = sky_service.get_wandb_run_name_from_sky_job("sky-job-id")

# Access metrics (requires wandb API key)
wandb_service = get_wandb_service()
runs = wandb_service.get_runs_for_experiment("experiment_name")
```

## Directory Structure

```
experiments/
├── experiment.py             # Base Experiment and ExperimentConfig classes
├── training_job.py          # TrainingJob wrapper and config
├── training_run_config.py   # YAML serialization logic
├── runner.py                # Typer CLI integration
├── recipes/                 # Concrete experiment implementations
│   └── arena_experiment.py  # SingleJobExperimentConfig example
└── services/                # Skypilot and WandB integration
```

## Implementation Notes

- Configs are saved to `configs/experiments/` (gitignored) with relative path defaults (`../agent/fast`)
- The `instance_name` (name + timestamp) ensures unique configs for each launch
- Skypilot's `file_mounts` transfers the YAML to remote, then copies it to `configs/experiments/`
- Remote execution uses `+experiments={instance_name}` to load the config
- All trainer fields must be provided when creating custom `TrainerConfig` (no partial overrides)
