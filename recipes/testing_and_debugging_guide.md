# Testing and Debugging Guide for Learning Progress Arena Experiment

## Overview

This guide helps you test the learning progress arena experiment runs and diagnose why they might be failing and recovering frequently.

## Testing the Runs

### 1. Local Testing (Small Scale)

Before running the full experiment, test locally with reduced parameters:

```bash
# Test learning progress curriculum locally
./devops/skypilot/launch.py train \
    --gpus=1 \
    --nodes=1 \
    --no-spot \
    run="test.learning_progress.local" \
    --config configs/user/learning_progress_experiment.yaml \
    trainer.total_timesteps=10_000_000 \
    trainer.num_workers=2 \
    trainer.batch_size=16384

# Test random curriculum locally
./devops/skypilot/launch.py train \
    --gpus=1 \
    --nodes=1 \
    --no-spot \
    run="test.random.local" \
    --config configs/user/random_curriculum_experiment.yaml \
    trainer.total_timesteps=10_000_000 \
    trainer.num_workers=2 \
    trainer.batch_size=16384

# Test basic arena locally
./devops/skypilot/launch.py train \
    --gpus=1 \
    --nodes=1 \
    --no-spot \
    run="test.basic.local" \
    --config configs/user/basic_arena_experiment.yaml \
    trainer.total_timesteps=10_000_000 \
    trainer.num_workers=2 \
    trainer.batch_size=16384
```

### 2. Monitor Run Status

```bash
# Check wandb for run status
wandb runs list --project learning-progress-sweep --entity metta-research

# Check specific run details
wandb runs list --project learning-progress-sweep --entity metta-research | grep "your_run_name"
```

### 3. Check Logs and Metrics

Monitor these key metrics in wandb:

**Learning Progress Curriculum:**
- `lp/num_active_tasks` - Should be around 16
- `lp/mean_sample_prob` - Should be > 0
- `lp/task_success_rate` - Should improve over time
- `lp/mean_evals_per_task` - Should be reasonable

**All Runs:**
- `reward` - Should be increasing
- `loss/value_loss` - Should be decreasing
- `loss/policy_loss` - Should be stable
- `train/learning_rate` - Should follow schedule

## Common Failure Modes and Debugging

### 1. Memory Issues

**Symptoms:**
- OOM (Out of Memory) errors
- Runs restarting frequently
- GPU memory usage spikes

**Debugging:**
```bash
# Check GPU memory usage
nvidia-smi

# Monitor memory in wandb
# Look for memory-related metrics in the run logs
```

**Solutions:**
- Reduce `trainer.batch_size` (currently 524288)
- Reduce `trainer.num_workers` (currently 4)
- Increase `trainer.minibatch_size` to reduce memory fragmentation
- Add memory monitoring to the training script

### 2. Learning Progress Curriculum Issues

**Symptoms:**
- All tasks have zero learning progress
- Curriculum gets stuck on same tasks
- No adaptation happening

**Debugging:**
```python
# Check learning progress implementation
# In mettagrid/src/metta/mettagrid/curriculum/learning_progress.py:

# Look for these issues:
# 1. Division by zero in _reweight function
# 2. NaN values in task success rates
# 3. All learning progress values being zero
```

**Solutions:**
- Adjust hyperparameters:
  - Increase `ema_timescale` (currently 0.001)
  - Decrease `progress_smoothing` (currently 0.05)
  - Increase `sample_threshold` (currently 10)
- Add NaN checks in the learning progress calculation
- Ensure task success rates are properly normalized

### 3. Environment/Curriculum Issues

**Symptoms:**
- Environment crashes
- Invalid actions
- Curriculum not sampling tasks correctly

**Debugging:**
```bash
# Test individual arena environments
python -c "
from metta.mettagrid.mettagrid_env import MettaGridEnv
from omegaconf import OmegaConf

# Test basic_easy environment
cfg = OmegaConf.load('configs/env/mettagrid/arena/basic_easy.yaml')
env = MettaGridEnv(cfg)
obs = env.reset()
print('Environment reset successful')

# Test curriculum sampling
from metta.mettagrid.curriculum.learning_progress import LearningProgressCurriculum
curriculum = LearningProgressCurriculum({'task1': 1.0})
task = curriculum.get_task()
print('Curriculum sampling successful')
"
```

### 4. Checkpoint/Recovery Issues

**Symptoms:**
- Runs restarting from beginning
- Checkpoint corruption
- State not being restored properly

**Debugging:**
```bash
# Check checkpoint files
ls -la train_dir/*/checkpoints/

# Verify checkpoint integrity
python -c "
import torch
checkpoint = torch.load('train_dir/run_name/checkpoints/trainer_state.pt')
print('Checkpoint keys:', checkpoint.keys())
print('Agent step:', checkpoint['agent_step'])
print('Epoch:', checkpoint['epoch'])
"
```

**Solutions:**
- Increase checkpoint frequency: `trainer.checkpoint.checkpoint_interval: 25`
- Add checkpoint validation
- Implement checkpoint corruption detection

### 5. Hyperparameter Issues

**Symptoms:**
- Training not converging
- Loss exploding
- Poor performance

**Debugging:**
```bash
# Test different hyperparameter combinations
./devops/skypilot/launch.py train \
    --gpus=1 \
    --nodes=1 \
    --no-spot \
    run="test.hyperparams" \
    --config configs/user/learning_progress_experiment.yaml \
    trainer.total_timesteps=10_000_000 \
    ema_timescale=0.01 \
    progress_smoothing=0.1 \
    num_active_tasks=8 \
    rand_task_rate=0.5
```

## Specific Testing Scripts

### 1. Quick Health Check

```bash
#!/bin/bash
# quick_health_check.sh

echo "=== Learning Progress Arena Health Check ==="

# Test environment loading
echo "Testing environment loading..."
python -c "
from metta.mettagrid.mettagrid_env import MettaGridEnv
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/env/mettagrid/arena/basic_easy.yaml')
env = MettaGridEnv(cfg)
print('✓ Environment loads successfully')
"

# Test curriculum loading
echo "Testing curriculum loading..."
python -c "
from metta.mettagrid.curriculum.learning_progress import LearningProgressCurriculum
curriculum = LearningProgressCurriculum({'task1': 1.0})
print('✓ Curriculum loads successfully')
"

# Test configuration loading
echo "Testing configuration loading..."
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/user/learning_progress_experiment.yaml')
print('✓ Configuration loads successfully')
print(f'  - Total timesteps: {cfg.trainer.total_timesteps}')
print(f'  - Num workers: {cfg.trainer.num_workers}')
print(f'  - Batch size: {cfg.trainer.batch_size}')
"
```

### 2. Memory Usage Test

```bash
#!/bin/bash
# memory_test.sh

echo "=== Memory Usage Test ==="

# Monitor GPU memory during short training run
nvidia-smi dmon -s pucvmet -d 1 &
SMI_PID=$!

./devops/skypilot/launch.py train \
    --gpus=1 \
    --nodes=1 \
    --no-spot \
    run="memory.test" \
    --config configs/user/learning_progress_experiment.yaml \
    trainer.total_timesteps=1_000_000 \
    trainer.num_workers=2 \
    trainer.batch_size=16384

kill $SMI_PID
```

### 3. Curriculum Behavior Test

```bash
#!/bin/bash
# curriculum_test.sh

echo "=== Curriculum Behavior Test ==="

# Test learning progress curriculum with logging
python -c "
import logging
logging.basicConfig(level=logging.INFO)

from metta.mettagrid.curriculum.learning_progress import LearningProgressCurriculum
import numpy as np

# Create curriculum with test tasks
tasks = {f'task_{i}': 1.0 for i in range(12)}
curriculum = LearningProgressCurriculum(
    tasks=tasks,
    ema_timescale=0.001,
    progress_smoothing=0.05,
    num_active_tasks=16,
    rand_task_rate=0.25,
    sample_threshold=10,
    memory=25
)

print('Testing curriculum sampling...')
for i in range(100):
    task = curriculum.get_task()
    # Simulate task completion with random score
    score = np.random.random()
    curriculum.complete_task(task.id, score)

    if i % 20 == 0:
        stats = curriculum.stats()
        print(f'Step {i}: {stats}')
"
```

## Recovery Strategies

### 1. Automatic Restart with Backoff

```bash
#!/bin/bash
# restart_with_backoff.sh

MAX_RETRIES=5
BACKOFF_SECONDS=60

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "Attempt $attempt/$MAX_RETRIES"

    ./devops/skypilot/launch.py train \
        --gpus=1 \
        --nodes=1 \
        --no-spot \
        run="test.with.restart" \
        --config configs/user/learning_progress_experiment.yaml \
        trainer.total_timesteps=10_000_000

    if [ $? -eq 0 ]; then
        echo "Training completed successfully"
        break
    else
        echo "Training failed, waiting $BACKOFF_SECONDS seconds before retry..."
        sleep $BACKOFF_SECONDS
        BACKOFF_SECONDS=$((BACKOFF_SECONDS * 2))
    fi
done
```

### 2. Checkpoint Recovery

```bash
#!/bin/bash
# checkpoint_recovery.sh

RUN_NAME="your_run_name"
CHECKPOINT_DIR="train_dir/$RUN_NAME/checkpoints"

# Check if checkpoint exists
if [ -f "$CHECKPOINT_DIR/trainer_state.pt" ]; then
    echo "Found checkpoint, resuming training..."
    ./devops/skypilot/launch.py train \
        --gpus=1 \
        --nodes=1 \
        --no-spot \
        run="$RUN_NAME" \
        --config configs/user/learning_progress_experiment.yaml \
        trainer.total_timesteps=1_000_000_000
else
    echo "No checkpoint found, starting fresh..."
    ./devops/skypilot/launch.py train \
        --gpus=1 \
        --nodes=1 \
        --no-spot \
        run="$RUN_NAME" \
        --config configs/user/learning_progress_experiment.yaml \
        trainer.total_timesteps=1_000_000_000
fi
```

## Monitoring and Alerting

### 1. Key Metrics to Monitor

```python
# wandb_metrics.py
import wandb

def monitor_key_metrics():
    """Monitor key metrics for learning progress experiment"""
    metrics = {
        # Training progress
        'train/learning_rate': 'Should follow schedule',
        'train/entropy_loss': 'Should be stable',
        'train/value_loss': 'Should decrease',
        'train/policy_loss': 'Should be stable',

        # Learning progress specific
        'lp/num_active_tasks': 'Should be around 16',
        'lp/mean_sample_prob': 'Should be > 0',
        'lp/task_success_rate': 'Should improve',
        'lp/mean_evals_per_task': 'Should be reasonable',

        # Environment performance
        'reward': 'Should be increasing',
        'episode_length': 'Should be stable',
        'episode_reward': 'Should be increasing',

        # System metrics
        'system/gpu_memory_used': 'Should be stable',
        'system/cpu_usage': 'Should be reasonable',
    }

    return metrics
```

### 2. Failure Detection

```python
# failure_detection.py
def detect_failures(run_logs):
    """Detect common failure patterns"""
    failures = []

    # Check for OOM errors
    if "CUDA out of memory" in run_logs:
        failures.append("GPU OOM - reduce batch size")

    # Check for NaN losses
    if "loss contains NaN" in run_logs:
        failures.append("NaN loss - check learning rate")

    # Check for curriculum issues
    if "lp/num_active_tasks" == 0:
        failures.append("Learning progress not working")

    # Check for checkpoint issues
    if "Failed to load checkpoint" in run_logs:
        failures.append("Checkpoint corruption")

    return failures
```

## Summary

The most likely causes of frequent failures and recoveries are:

1. **Memory Issues**: Large batch sizes and complex environments
2. **Learning Progress Algorithm**: Division by zero, NaN values
3. **Checkpoint Corruption**: Network issues during save/load
4. **Environment Complexity**: 12 different arena tasks with varying difficulty
5. **Hyperparameter Sensitivity**: Learning progress parameters need tuning

**Recommended Testing Order:**
1. Run local small-scale tests
2. Monitor memory usage
3. Test curriculum behavior
4. Check checkpoint integrity
5. Validate hyperparameters
6. Implement proper monitoring and recovery

This should help you identify and resolve the failure/recovery issues in your learning progress arena experiment.
