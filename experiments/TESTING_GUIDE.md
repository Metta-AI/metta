# Experiments Testing Guide

Quick reference for testing the experiments framework locally and debugging common issues.

## Local Testing Workflow

### 1. Preview Without Launching

```bash
# Generate config and see what will be launched
uv run experiments/recipes/arena_experiment.py --no-launch

# Output includes:
# - YAML path: configs/experiments/arena_experiment_20250814_100240.yaml
# - Test command: uv run ./tools/train.py +experiments=arena_experiment_20250814_100240 run=arena_experiment_20250814_100240
```

### 2. Test Generated Config Locally

```bash
# Use the command from preview output (note: run parameter is required)
uv run ./tools/train.py +experiments=arena_experiment_20250814_100240 \
  run=arena_experiment_20250814_100240 \
  trainer.total_timesteps=1000 \
  trainer.simulation.skip_git_check=true \
  wandb=off

# View the composed config
uv run ./tools/train.py +experiments=arena_experiment_20250814_100240 \
  run=test --cfg job
```

### 3. Launch to Skypilot

```bash
# Launch with defaults
uv run experiments/recipes/arena_experiment.py

# Custom infrastructure
uv run experiments/recipes/arena_experiment.py --gpus=4 --nodes=2 --spot=false
```

## Config Verification

### Generated YAML Structure

Configs are saved to `configs/experiments/` with relative path defaults:

```yaml
# @package _global_
defaults:
- ../common                    # Relative paths from experiments/
- ../agent/fast
- ../trainer/trainer
- ../sim/arena
- ../wandb/metta_research
- _self_

trainer:
  curriculum: env/mettagrid/curriculum/arena/learning_progress
  total_timesteps: 10000000
  # ... full trainer config
```

### Skypilot Transfer Verification

```bash
# Check job status
sky jobs queue

# Verify config was transferred
sky exec JOB_ID 'ls -la /tmp/metta_train_config.yaml'

# View running command
sky exec JOB_ID 'ps aux | grep train.py'
```

## Unit Tests

```bash
# Run all experiment tests
uv run pytest tests/experiments/ -xvs

# Specific test suites
uv run pytest tests/experiments/test_yaml_contract.py -xvs      # YAML structure
uv run pytest tests/experiments/test_train_integration.py -xvs  # Hydra loading
uv run pytest tests/experiments/test_experiment_workflows.py -xvs # E2E workflows
```

## Quick Debug Commands

```python
# Test config creation
uv run python -c "
from experiments.recipes.arena_experiment import ArenaExperimentConfig
from experiments.experiment import SingleJobExperiment

config = ArenaExperimentConfig(name='debug_test', launch=False)
experiment = SingleJobExperiment(config)
experiment.load_or_launch_training_jobs()
print(f'Generated: {experiment.instance_name}')
"

# Verify YAML serialization
uv run python -c "
from experiments.training_run_config import TrainingRunConfig
config = TrainingRunConfig(curriculum='test/curriculum')
path, yaml_dict = config.serialize_to_yaml_file('test_instance')
print(f'YAML at: {path}')
print(f'Defaults: {yaml_dict[\"defaults\"]}')
"
```

## Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `Missing mandatory value: run` | Hydra requires run parameter | Add `run=your_run_name` to command |
| `Field required: curriculum` | TrainingRunConfig needs curriculum | Ensure experiment sets `curriculum` field |
| `Git uncommitted changes` | Git safety check for remote execution | Add `trainer.simulation.skip_git_check=true` for local testing |
| `ValidationError: minibatch_size` | Batch size validation | Ensure `minibatch_size <= batch_size` and divides evenly |
| `checkpoint_dir must be set` | TrainerConfig validation | Provide all required fields when creating TrainerConfig |

## Creating Custom Experiments

### Single Job Experiment
```python
# experiments/recipes/my_experiment.py
from experiments.experiment import SingleJobExperiment, SingleJobExperimentConfig
from experiments.training_run_config import TrainingRunConfig

class MyExperimentConfig(SingleJobExperimentConfig):
    name: str = "my_experiment"
    training: TrainingRunConfig = TrainingRunConfig(
        curriculum="env/mettagrid/curriculum/navigation/basic",
        agent_config="latent_attn_tiny",  # References configs/agent/
        wandb_tags=["navigation", "test"],
    )

if __name__ == "__main__":
    from experiments.runner import runner
    runner(SingleJobExperiment, MyExperimentConfig)
```

### Multi-Job A/B Test
```python
# experiments/recipes/lr_comparison.py
from typing import List
from experiments.experiment import Experiment, ExperimentConfig
from experiments.training_job import TrainingJobConfig
from experiments.training_run_config import TrainingRunConfig
from experiments.skypilot_job_config import SkypilotJobConfig
from metta.rl.trainer_config import (
    TrainerConfig, OptimizerConfig, CheckpointConfig,
    SimulationConfig, TorchProfilerConfig
)

class LRComparisonExperiment(Experiment):
    """Compare different learning rates."""
    
    def training_job_configs(self) -> List[TrainingJobConfig]:
        configs = []
        for lr in [0.0001, 0.0003]:
            trainer = TrainerConfig(
                num_workers=4,
                optimizer=OptimizerConfig(learning_rate=lr),
                checkpoint=CheckpointConfig(checkpoint_dir="${run_dir}/checkpoints"),
                simulation=SimulationConfig(replay_dir="${run_dir}/replays"),
                profiler=TorchProfilerConfig(profile_dir="${run_dir}/torch_traces"),
            )
            training = TrainingRunConfig(
                curriculum="env/mettagrid/curriculum/arena/basic",
                trainer=trainer,
                wandb_tags=["lr_comparison", f"lr_{lr}"],
            )
            configs.append(TrainingJobConfig(
                skypilot=SkypilotJobConfig(gpus=1),
                training=training,
            ))
        return configs

class LRComparisonConfig(ExperimentConfig):
    name: str = "lr_comparison"

if __name__ == "__main__":
    from experiments.runner import runner
    runner(LRComparisonExperiment, LRComparisonConfig)
```

## Implementation Notes

- Configs saved to `configs/experiments/` are gitignored
- Instance names include timestamp: `{name}_{YYYYMMDD_HHMMSS}`
- Skypilot mounts config to `/tmp/metta_train_config.yaml`
- Remote execution uses `--config-path=/tmp --config-name=metta_train_config`
- All TrainerConfig fields must be provided (no partial updates)