# Experiments Framework

A Python-based framework for defining, launching, and managing distributed training experiments on Skypilot.

## Overview

The experiments framework provides:
- **Type-safe configuration** using Pydantic models that serialize to Hydra-compatible YAML
- **Clean separation** between infrastructure (Skypilot) and training (Hydra) configuration
- **File-based config transfer** via Skypilot's file_mounts (no git commits required)
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

Create a new experiment by extending `SingleJobExperimentConfig`:

```python
# experiments/recipes/my_ab_test.py
from experiments.experiment import SingleJobExperiment, SingleJobExperimentConfig
from experiments.training_run_config import TrainingRunConfig

class MyABTestConfig(SingleJobExperimentConfig):
    """A/B test comparing two learning rates."""
    
    name: str = "lr_ab_test"
    training: TrainingRunConfig = TrainingRunConfig(
        curriculum="env/mettagrid/curriculum/arena/learning_progress",
        wandb_tags=["ab_test", "learning_rate"],
    )

if __name__ == "__main__":
    from experiments.runner import runner
    runner(SingleJobExperiment, MyABTestConfig)
```

## Architecture

### Configuration Hierarchy

```python
TrainingJobConfig
├── SkypilotJobConfig      # Infrastructure: GPUs, nodes, spot instances
└── TrainingRunConfig       # Training: agent, curriculum, hyperparameters
    └── TrainerConfig       # Nested: optimizer, PPO, checkpointing
```

### How Config Transfer Works

1. **Local**: Python config objects are created and validated
2. **Serialization**: `TrainingRunConfig.serialize_to_yaml_file()` creates a Hydra-compatible YAML
3. **Transfer**: Skypilot mounts the YAML to `/tmp/metta_train_config.yaml` on remote
4. **Execution**: Remote training loads via `--config-path=/tmp --config-name=metta_train_config`

Example serialized YAML structure:
```yaml
defaults:
  - ../common
  - ../agent/fast
  - ../trainer/trainer
  - ../sim/arena
  - ../wandb/metta_research
  - _self_

trainer:
  curriculum: env/mettagrid/curriculum/arena/learning_progress
  total_timesteps: 10000000
  # ... nested trainer config
```

## Advanced Usage

### Custom Training Configuration

For experiments requiring specific hyperparameters:

```python
from metta.rl.trainer_config import TrainerConfig, OptimizerConfig

trainer = TrainerConfig(
    total_timesteps=50_000_000,
    batch_size=8192,
    minibatch_size=256,
    num_workers=8,
    optimizer=OptimizerConfig(
        type="muon",
        learning_rate=0.001,
    ),
    checkpoint=CheckpointConfig(checkpoint_dir="${run_dir}/checkpoints"),
    simulation=SimulationConfig(replay_dir="${run_dir}/replays"),
    profiler=TorchProfilerConfig(profile_dir="${run_dir}/torch_traces"),
)

training = TrainingRunConfig(
    curriculum="your/curriculum/path",
    trainer=trainer,
)
```

### Extending for Sweep Capabilities

The framework is designed for extension. To add sweep support:

1. Create a `SweepExperimentConfig` that generates multiple `TrainingJobConfig` instances
2. Override `training_job_configs()` to return the sweep configurations
3. The existing launch infrastructure handles multiple jobs automatically

```python
class SweepExperiment(Experiment):
    def training_job_configs(self) -> List[TrainingJobConfig]:
        configs = []
        for lr in [0.0001, 0.0003, 0.001]:
            trainer = TrainerConfig(learning_rate=lr, ...)
            configs.append(TrainingJobConfig(training=TrainingRunConfig(trainer=trainer)))
        return configs
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
└── services/                # Skypilot and WandB integration
```

## Implementation Notes

- Configs are saved to `configs/experiments/` (gitignored) with relative path defaults (`../agent/fast`)
- The `instance_name` (name + timestamp) ensures unique configs for each launch
- Skypilot's `file_mounts` transfers configs without requiring git commits
- All trainer fields must be provided when creating custom `TrainerConfig` (no partial overrides)