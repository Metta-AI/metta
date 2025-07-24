# Dual-Policy Training with Checkpoint NPCs

This guide explains how to use the dual-policy training system with checkpoint-based NPCs (Non-Player Characters) instead of scripted behaviors.

## Overview

The dual-policy system allows you to train a new policy against an old policy checkpoint. This is useful for:

- **Self-play training**: Training against previous versions of your own policy
- **Adversarial training**: Training against a specific opponent policy
- **Curriculum learning**: Gradually increasing difficulty by training against stronger policies
- **Evaluation**: Testing how well a new policy performs against known baselines

## Configuration

### Basic Configuration

Create a configuration file that extends the dual-policy system:

```yaml
# configs/user/my_dual_policy_checkpoint.yaml
defaults:
  - /common
  - /agent/fast
  - /trainer/trainer
  - /sim/all@evals
  - _self_

run: my_dual_policy_experiment
cmd: train

trainer:
  dual_policy:
    enabled: true
    policy_a_percentage: 0.5  # 50% main policy, 50% NPC

    npc_type: "checkpoint"
    checkpoint_npc:
      checkpoint_path: "wandb://metta-research/metta/model/your_old_policy_run:latest"
```

### Configuration Options

#### `checkpoint_path`
- **Type**: `str`
- **Description**: Path to the checkpoint file to use as the NPC
- **Supported formats**:
  - **WandB URI**: `"wandb://metta-research/metta/model/run_name:latest"`
  - **Local file**: `"./train_dir/old_run/checkpoints/model_0000.pt"`
  - **S3 URI**: `"s3://bucket/path/to/checkpoint.pt"`

#### `policy_a_percentage`
- **Type**: `float` (0.0 to 1.0)
- **Default**: `0.5`
- **Description**: Percentage of agents controlled by the main policy (the rest use the NPC)

## Finding Available WandB Policies

### Using the WandB Run Lister

The easiest way to find available policies is to use the built-in tool:

```bash
# List recent finished runs
python tools/list_wandb_runs.py --limit 10

# List only the URIs for easy copying
python tools/list_wandb_runs.py --format uris --limit 5

# List runs in JSON format for detailed info
python tools/list_wandb_runs.py --format json --limit 3
```

Example output:
```
Run Name                                           Score      Created      WandB URI
------------------------------------------------------------------------------------------------------------
yudhister.recipes.arena.2x8.efficiency_baseline.07-24-00-18 N/A        2025-07-24   wandb://metta-research/metta/model/yudhister.recipes.arena.2x8.efficiency_baseline.07-24-00-18:latest
yudhister.recipes.arena.4x4.efficiency_baseline.07-24-00-17 N/A        2025-07-24   wandb://metta-research/metta/model/yudhister.recipes.arena.4x4.efficiency_baseline.07-24-00-17:latest
```

### Manual WandB API Access

You can also query WandB directly:

```python
import wandb
api = wandb.Api()
runs = api.runs('metta-research/metta', filters={'state': 'finished'}, order='-created_at')
for run in runs[:5]:
    print(f"{run.name}: wandb://metta-research/metta/model/{run.name}:latest")
```

## Usage Examples

### 1. Using the Recipe Script

The easiest way to run dual-policy training with a WandB checkpoint:

```bash
# Basic usage (uses default WandB policy)
./recipes/dual_policy_checkpoint.sh

# With a specific WandB policy
./recipes/dual_policy_checkpoint.sh \
  "wandb://metta-research/metta/model/yudhister.recipes.arena.2x8.efficiency_baseline.07-24-00-18:latest" \
  5000000000 \
  8

# With a different successful policy
./recipes/dual_policy_checkpoint.sh \
  "wandb://metta-research/metta/model/yudhister.recipes.arena.4x4.efficiency_baseline.07-24-00-17:latest"
```

### 2. Direct Command Line

```bash
# Set cost tracking
export METTA_HOURLY_COST=10.0

# Run training with WandB policy
python tools/train.py \
  +user=dual_policy_checkpoint_example \
  run=my_experiment \
  trainer.dual_policy.checkpoint_npc.checkpoint_path="wandb://metta-research/metta/model/yudhister.recipes.arena.2x8.efficiency_baseline.07-24-00-18:latest" \
  trainer.total_timesteps=10000000000 \
  trainer.num_workers=4
```

### 3. Local Testing

For local testing with a local checkpoint:

```bash
python tools/train.py \
  +user=dual_policy_checkpoint_local_example \
  run=local_test \
  trainer.dual_policy.checkpoint_npc.checkpoint_path="./train_dir/previous_run/checkpoints/model_0000.pt" \
  trainer.total_timesteps=1000000 \
  trainer.num_workers=2 \
  wandb=off
```

## Finding Good WandB Policies

To find good policies to use as NPCs, you can:

### 1. Use Recent Successful Runs
Look for recent training runs that have good performance:

```bash
# Example WandB URIs for good policies:
"wandb://metta-research/metta/model/yudhister.recipes.arena.2x8.efficiency_baseline.07-24-00-18:latest"
"wandb://metta-research/metta/model/yudhister.recipes.arena.4x4.efficiency_baseline.07-24-00-17:latest"
"wandb://metta-research/metta/model/yudhister.recipes.arena.2x4.efficiency_baseline.07-24-00-17:latest"
```

### 2. Check WandB Dashboard
Visit the [Metta WandB project](https://wandb.ai/metta-research/metta) to browse:
- Recent successful runs
- High-scoring policies
- Different training approaches

### 3. Use Specific Policy Versions
You can specify exact versions instead of "latest":
```bash
"wandb://metta-research/metta/model/run_name:v1"
"wandb://metta-research/metta/model/run_name:v2"
```

## Monitoring and Metrics

When dual-policy training is enabled, the following metrics are logged to WandB:

### Reward Metrics
- `dual_policy/policy_a_reward` - Average reward for main policy agents
- `dual_policy/policy_a_reward_total` - Total reward for main policy agents
- `dual_policy/npc_reward` - Average reward for NPC agents
- `dual_policy/npc_reward_total` - Total reward for NPC agents
- `dual_policy/combined_reward` - Average combined reward
- `dual_policy/combined_reward_total` - Total combined reward

### Agent Count Metrics
- `dual_policy/policy_a_agent_count` - Number of main policy agents
- `dual_policy/npc_agent_count` - Number of NPC agents

## Checkpoint Format Support

The system supports various checkpoint formats:

### WandB Artifacts
```python
# WandB artifact format
"wandb://entity/project/artifact_type/name:version"
"wandb://metta-research/metta/model/run_name:latest"
```

### PyTorch Checkpoints
```python
# Standard PyTorch checkpoint
checkpoint = torch.load("model.pt")
policy = checkpoint['model']  # or checkpoint['policy']
```

### State Dicts
```python
# State dictionary
checkpoint = torch.load("state_dict.pt")
policy = checkpoint  # Direct state dict
```

### Custom Formats
The loader automatically detects common checkpoint structures and extracts the policy component.

## Troubleshooting

### Common Issues

1. **WandB artifact not found**
   ```
   Error: Could not load NPC policy from WandB artifact
   ```
   - Verify the WandB URI format is correct
   - Ensure the artifact exists and is accessible
   - Check that you have access to the WandB project
   - **Use the WandB run lister tool**: `python tools/list_wandb_runs.py` to find valid run names

2. **Policy loading errors**
   ```
   Error: Checkpoint NPC policy must be a PyTorch module
   ```
   - Ensure the checkpoint contains a valid PyTorch model
   - Check that the checkpoint format is supported

3. **Device mismatch**
   ```
   Error: Expected device cuda:0 but found cpu
   ```
   - The system automatically handles device placement
   - Check that the checkpoint was saved on a compatible device

### Debug Tips

1. **Enable debug logging**:
   ```bash
   export LOG_LEVEL=DEBUG
   ```

2. **Test WandB artifact loading**:
   ```python
   from metta.rl.util.policy_loader import load_policy_from_checkpoint
   import torch

   policy = load_policy_from_checkpoint("wandb://metta-research/metta/model/run_name:latest", torch.device("cpu"))
   print(f"Policy loaded: {type(policy)}")
   ```

3. **Verify dual-policy initialization**:
   Look for this log message:
   ```
   INFO: Loaded NPC policy from WandB artifact: entity/project/artifact_type/name:version
   INFO: Dual-policy system initialized: {...}
   ```

4. **List available runs**:
   ```bash
   python tools/list_wandb_runs.py --limit 10
   ```

## Advanced Usage

### Multiple Checkpoints

To train against multiple checkpoints, you can:

1. **Use different runs**: Train separate experiments with different checkpoints
2. **Curriculum approach**: Start with weak checkpoints, gradually increase difficulty
3. **Ensemble approach**: Create a custom NPC that randomly selects from multiple checkpoints

### Custom NPC Logic

For more complex scenarios, you can extend the system:

1. **Modify the policy loader** to support custom checkpoint formats
2. **Add ensemble logic** to select from multiple checkpoints
3. **Implement adaptive difficulty** based on training progress

## Performance Considerations

- **Memory usage**: Loading an additional policy increases memory requirements
- **Computation overhead**: NPC inference adds computational cost
- **Checkpoint size**: Large checkpoints may slow down initialization
- **WandB bandwidth**: Downloading artifacts from WandB adds network overhead

## Best Practices

1. **Use WandB artifacts** for production runs to avoid local storage issues
2. **Monitor memory usage** when using large policies
3. **Start with shorter runs** to verify configuration
4. **Use meaningful run names** for easy identification
5. **Enable cost tracking** to monitor resource usage
6. **Test with local checkpoints** first before using WandB artifacts
7. **Use the WandB run lister tool** to find valid run names instead of guessing
