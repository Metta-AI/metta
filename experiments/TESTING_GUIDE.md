# Testing Guide for Experiments Framework

This guide provides instructions for testing the experiments framework with YAML serialization for Skypilot jobs.

## Overview

The system works by:
1. Creating a `TrainingRunConfig` with all training parameters
2. Serializing it to a temporary YAML file with complete Hydra-compatible configuration
3. Transferring the YAML via Skypilot's file_mounts to `/tmp/metta_train_config.yaml`
4. Loading the config on the remote using Hydra's `--config-path=/tmp --config-name=metta_train_config`

## Quick Test Commands

### 1. Preview Mode (No Launch)

Test locally without launching to see what will be generated:

```bash
# Basic preview - shows YAML without launching
uv run experiments/runner.py single --name test_experiment --no-launch

# Preview with custom parameters
uv run experiments/runner.py single \
  --name test_experiment \
  --no-launch \
  --gpus 4 \
  --nodes 2 \
  --total-timesteps 1000000 \
  --learning-rate 0.0003
```

This will:
- Generate the YAML configuration
- Display the config details that will be sent
- Show local testing commands with `tools/train.py`
- Create YAML files in `/tmp/metta_configs/` and `~/.metta_test_configs/`
- NOT launch to Skypilot

### 2. Test Local YAML with tools/train.py

The preview mode provides a command to test locally:

```bash
# Run preview mode first
uv run experiments/runner.py single --name test_local --no-launch

# Look for output like:
# To test locally with tools/train.py:
#   uv run ./tools/train.py --config-path=/Users/you/.metta_test_configs --config-name=test_config_20250113_123456

# Copy and run that command to test the YAML locally
uv run ./tools/train.py --config-path=/Users/you/.metta_test_configs --config-name=test_config_20250113_123456 --cfg job
```

### 3. Full Launch to Skypilot

When ready to launch:

```bash
# Basic launch
uv run experiments/runner.py single --name my_training_run

# With custom settings
uv run experiments/runner.py single \
  --name my_training_run \
  --gpus 2 \
  --total-timesteps 500000 \
  --batch-size 1024 \
  --curriculum "env/mettagrid/curriculum/arena/learning_progress"
```

## Verifying YAML Structure

### Check Generated YAML

After running in preview mode, examine the generated YAML:

```bash
# Find YAML files
ls -la /tmp/metta_configs/
ls -la ~/.metta_test_configs/

# Inspect the content
cat ~/.metta_test_configs/test_config_*.yaml
```

The YAML should contain:
- `defaults:` section with Hydra config composition
- `trainer:` section with all training parameters
- `seed:`, `py_agent:`, `train_job:` and other required fields
- No `# @package _global_` directive (that's only for local testing files)

### Verify with Hydra

Test that Hydra can load the config:

```bash
# This shows the composed config without running training
uv run ./tools/train.py \
  --config-path=/path/to/yaml/dir \
  --config-name=config_name \
  --cfg job \
  hydra.mode=MULTIRUN \
  hydra.dry=true
```

## Checking Skypilot Logs

After launching, monitor the job:

```bash
# Check job status
sky jobs queue

# View logs (replace JOB_ID with actual ID)
sky jobs logs JOB_ID

# Check the transferred config file
sky exec JOB_ID 'cat /tmp/metta_train_config.yaml'

# Check that tools/train.py received the config correctly
sky exec JOB_ID 'grep "config-path=/tmp" /tmp/sky_logs/*.log'
```

## Troubleshooting

### Common Issues

1. **"Could not load 'wandb: metta_research'"**
   - This happens when testing locally with custom config paths
   - The YAML references configs that aren't in the custom path
   - This is expected; the remote has all configs available

2. **FileNotFoundError for YAML**
   - Check that the path shown in preview mode exists
   - Ensure `/tmp/metta_configs/` or `~/.metta_test_configs/` directories are writable

3. **Skypilot launch fails**
   - Check git state: `git status` (should be clean)
   - Verify AWS/wandb credentials are set
   - Check that launch.py has --config-file support

### Debug Commands

```bash
# Check that TrainingRunConfig serialization works
uv run python -c "
from experiments.training_run_config import TrainingRunConfig
config = TrainingRunConfig()
path, yaml_dict = config.serialize_to_yaml_file()
print(f'Created: {path}')
print(f'Keys: {list(yaml_dict.keys())}')
"

# Test SingleJobExperiment config generation
uv run python -c "
from experiments.experiment import SingleJobExperimentConfig, SingleJobExperiment
config = SingleJobExperimentConfig(name='test', launch=False)
exp = SingleJobExperiment(config)
exp.load_or_launch_training_jobs()
"
```

## Running Tests

```bash
# Run all experiments tests
uv run pytest tests/experiments/ -v

# Run specific test suites
uv run pytest tests/experiments/test_yaml_contract.py -v  # YAML structure tests
uv run pytest tests/experiments/test_train_integration.py -v  # Integration with train.py
uv run pytest tests/experiments/test_experiment_workflows.py -v  # User workflow tests
```