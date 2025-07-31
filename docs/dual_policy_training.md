# Dual-Policy Training

This document describes the dual-policy training feature that allows training a new policy against an existing NPC policy loaded from a wandb URI.

## Overview

Dual-policy training enables scenarios where:
- P% of agents in each environment use the training policy (being learned)
- The remaining agents use an NPC policy (loaded from wandb URI)
- Only the training policy agents' experience is used for training updates
- The NPC policy agents act as part of the environment

## Configuration

### Basic Setup

Dual-policy training is disabled by default. To enable it, set `enabled: true` in your configuration:

```yaml
trainer:
  dual_policy:
    enabled: true  # Must be explicitly enabled
    training_agents_pct: 0.5  # 50% of agents use training policy
    checkpoint_npc:
      uri: "wandb://metta-research/dual_policy_training/model/bullm_dual_policy_against_roomba_v9:v2"
      type: "specific"  # Use specific version
      range: 1
      metric: "epoch"
```

### Parameters

- `enabled`: Enable/disable dual-policy training (default: false)
- `training_agents_pct`: Percentage of agents that use the training policy (0.0 to 1.0, default: 0.5)
- `checkpoint_npc`: Configuration for loading the NPC policy (same as `initial_policy`)

## Usage

### Using the Recipe Script

```bash
# Basic usage with default parameters
./recipes/dual_policy_sky.sh my_experiment_name

# Custom NPC policy URI
./recipes/dual_policy_sky.sh my_experiment_name "wandb://my_entity/my_project/model/my_npc_policy:v1"

# Custom timesteps and workers
./recipes/dual_policy_sky.sh my_experiment_name "wandb://my_uri" 5000000000 8
```

### Using SkyPilot Directly

```bash
./devops/skypilot/launch.py train \
    --gpus=1 \
    --nodes=1 \
    +user=dual_policy_checkpoint_example \
    run=my_experiment \
    trainer.dual_policy.enabled=true \
    trainer.dual_policy.checkpoint_npc.uri="wandb://my_uri" \
    trainer.dual_policy.training_agents_pct=0.6
```

## Implementation Details

### Agent Assignment

- Agents are assigned per environment using a simple rule
- First `training_agents_pct * num_agents_per_env` agents use the training policy
- Remaining agents use the NPC policy
- Assignment is fixed per environment (not random)

### Experience Collection

- Only training policy agents' experience is stored and used for training
- NPC policy agents' actions are sent to the environment but their experience is excluded
- This ensures the NPC policy acts as part of the environment rather than contributing to training

### Policy Loading

The NPC policy is loaded using the same mechanism as initial policies:
- Supports wandb URIs: `wandb://entity/project/artifact/name:version`
- Supports file URIs: `file://path/to/policy`
- Automatically initializes the policy to the environment

## Example Configuration

See `configs/user/dual_policy_checkpoint_example.yaml` for a complete example configuration.

## Testing

Run the dual-policy tests:

```bash
pytest tests/rl/test_dual_policy_training.py -v
```

## Limitations

- NPC policy must be compatible with the training environment
- Both policies must have the same observation and action spaces
- The feature is designed for training scenarios, not evaluation (use simulation for that)
