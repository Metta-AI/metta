# Complete Testing Guide for YAML Serialization

This guide provides step-by-step instructions for testing the YAML serialization system that transfers training configuration to Skypilot jobs.

## Overview

The system works by:
1. Creating a `TrainingRunConfig` with all training parameters
2. Serializing it to a temporary YAML file
3. Transferring the YAML via Skypilot's file_mounts to `/tmp/metta_train_config.yaml`
4. Loading the config on the remote using Hydra's `--config-path` and `--config-name`

## Test Commands

### 1. Basic Dry Run (Preview Mode)

First, test locally without launching to see the YAML that will be generated:

```bash
# Preview mode - shows YAML without launching
uv run experiments/recipes/arena_experiment.py --no-launch

# Preview with custom parameters
uv run experiments/recipes/arena_experiment.py \
  --no-launch \
  --gpus=4 \
  --nodes=2 \
  --total-timesteps=1000000 \
  --learning-rate=0.0003
```

This will:
- Generate the YAML configuration
- Display the full config that will be sent
- Show the file path where it's temporarily stored
- NOT launch to Skypilot

### 2. Test YAML Generation and Verification

Check that the YAML file is correctly generated:

```bash
# Run in preview mode and capture the YAML path
uv run experiments/recipes/arena_experiment.py --no-launch 2>&1 | grep "YAML config saved to"

# The output will show something like:
# YAML config saved to: /var/folders/.../metta_configs/train_config_20250113_123456_789012.yaml

# Inspect the generated YAML
cat /var/folders/.../metta_configs/train_config_20250113_123456_789012.yaml
```

Expected YAML structure:
```yaml
# @package _global_
agent: latent_attn_tiny
curriculum: env/mettagrid/curriculum/arena/learning_progress
wandb:
  entity: softmax-ai
  project: metta
  tags:
  - arena
  - experiments
trainer:
  total_timesteps: 10000000
  num_workers: 4
  batch_size: 4096
  # ... (full trainer config with all nested objects)
```

### 3. Launch with Skypilot (Real Test)

```bash
# Launch a real job
uv run experiments/recipes/arena_experiment.py \
  --gpus=1 \
  --nodes=1 \
  --total-timesteps=100000 \
  arena_test_$(date +%Y%m%d_%H%M%S)

# The command will:
# 1. Generate YAML config
# 2. Launch Skypilot job with the config mounted
# 3. Show the job name for tracking
```

### 4. Verify Config Transfer in Skypilot Logs

After launching, check that the config was transferred and loaded:

```bash
# Get the job name (shown in launch output)
JOB_NAME="arena_test_20250113_123456"

# Stream the logs
sky logs $JOB_NAME

# Or check status
sky status $JOB_NAME

# Look for these key indicators in the logs:
# 1. File mount confirmation:
#    "Mounting /tmp/metta_train_config.yaml"
# 
# 2. Hydra loading the config:
#    "Loading config from /tmp/metta_train_config.yaml"
#
# 3. Training starting with correct parameters:
#    "Starting training with config:"
#    "  total_timesteps: 100000"
#    "  agent: latent_attn_tiny"
```

### 5. SSH and Verify Config on Remote

For deeper inspection, SSH into the running job:

```bash
# SSH into the job
sky exec $JOB_NAME bash

# Once connected, verify:
# 1. Config file exists
ls -la /tmp/metta_train_config.yaml

# 2. Check contents
cat /tmp/metta_train_config.yaml

# 3. Check environment variables
env | grep METTA

# Expected output:
# METTA_RUN_ID=arena_test_20250113_123456
# METTA_CMD=train
# METTA_CMD_ARGS=--config-path=/tmp --config-name=metta_train_config
# METTA_CONFIG_FILE=/tmp/metta_train_config.yaml

# 4. Check that training process is using the config
ps aux | grep train.py
# Should show: python tools/train.py --config-path=/tmp --config-name=metta_train_config
```

### 6. Test Different Configurations

Test various parameter combinations:

```bash
# Minimal config
uv run experiments/recipes/arena_experiment.py minimal_test --no-launch

# Custom curriculum
uv run experiments/recipes/arena_experiment.py \
  --curriculum="env/mettagrid/curriculum/navigation/progressive" \
  --no-launch \
  nav_test

# Different agent architecture
uv run experiments/recipes/arena_experiment.py \
  --no-launch \
  --total-timesteps=5000000 \
  --learning-rate=0.001 \
  --batch-size=8192 \
  large_batch_test

# With specific wandb tags
uv run experiments/recipes/arena_experiment.py \
  --wandb-tags=experiment \
  --wandb-tags=yaml_test \
  --wandb-tags=v2 \
  --no-launch \
  tagged_test
```

### 7. Verify Hydra Integration

Test that the generated YAML works with Hydra directly:

```bash
# First generate a config
uv run experiments/recipes/arena_experiment.py --no-launch test_config

# Copy the shown YAML path
YAML_PATH="/var/folders/.../metta_configs/train_config_*.yaml"

# Copy to a test location
cp $YAML_PATH /tmp/test_config.yaml

# Test loading with Hydra (dry run)
uv run ./tools/train.py \
  --config-path=/tmp \
  --config-name=test_config \
  --cfg job \
  hydra.mode=MULTIRUN \
  hydra.dry=true

# This should show the merged configuration without actually running
```

### 8. Check Error Handling

Test various error conditions:

```bash
# Invalid parameters should fail with clear errors
uv run experiments/recipes/arena_experiment.py \
  --learning-rate=-1.0 \
  --no-launch \
  invalid_test

# Missing required fields (if any)
uv run experiments/recipes/arena_experiment.py \
  --no-launch \
  incomplete_test
```

## Common Issues and Solutions

### Issue 1: YAML file not found on remote
**Solution**: Check that file_mounts is working:
```bash
sky logs $JOB_NAME | grep "file_mounts"
```

### Issue 2: Hydra can't load config
**Solution**: Verify YAML syntax:
```bash
python -c "import yaml; yaml.safe_load(open('/tmp/metta_train_config.yaml'))"
```

### Issue 3: Training using wrong parameters
**Solution**: Check that METTA_CMD_ARGS is set correctly:
```bash
sky exec $JOB_NAME 'env | grep METTA_CMD_ARGS'
```

### Issue 4: Config overrides not applied
**Solution**: Ensure trainer overrides are properly nested in YAML

## Expected Success Indicators

✅ Preview mode shows complete YAML configuration
✅ YAML file is created in temp directory
✅ Skypilot mounts file to `/tmp/metta_train_config.yaml`
✅ Training starts with correct parameters from YAML
✅ WandB shows correct experiment tags and config
✅ No command-line length errors
✅ Config changes reflected in training behavior

## Advanced Testing

### Test with Multiple Jobs
```bash
# Launch multiple experiments with different configs
for i in {1..3}; do
  uv run experiments/recipes/arena_experiment.py \
    --learning-rate=0.000$i \
    --total-timesteps=$((1000000 * i)) \
    multi_test_$i
done
```

### Test Config Persistence
```bash
# Verify configs are saved and can be reused
ls -la /tmp/metta_configs/
# Each run creates a unique timestamped YAML
```

### Integration with WandB
```bash
# After job completes, check WandB for:
# 1. Correct config logged
# 2. Tags applied properly  
# 3. Hyperparameters tracked
```

## Summary

The YAML serialization system enables:
1. **Clean separation** between infrastructure (Skypilot) and training configs
2. **No command-line length limits** - complex configs transferred as files
3. **Type-safe configuration** with Pydantic validation
4. **Reproducible experiments** with versioned config files
5. **Easy debugging** - configs are human-readable YAML files