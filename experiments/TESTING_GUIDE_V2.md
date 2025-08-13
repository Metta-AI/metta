# Testing Guide V2 - YAML Serialization with In-Memory Config Updates

## Overview

The experiment system now follows this mental model:
1. **All configuration happens in-memory** - additional_args update the Python config objects directly
2. **YAML is fully definitive** - we serialize the complete config to YAML and minimize command-line args
3. **Agent configs are loaded locally** - we load agent YAML files locally to merge configurations

## Architecture Changes

### Previous Approach (Command-Line Heavy)
```
Config → Command-line args → Skypilot → Remote hydra composition
```

### New Approach (YAML-Definitive)
```
Config → Apply updates in-memory → Serialize complete YAML → Transfer via file_mounts
```

## Key Components

### 1. TrainingRunConfig
- Holds all training parameters
- `additional_args` now modify the config object directly via `apply_additional_args()`
- Serializes to complete, self-contained YAML

### 2. Agent Config Loading
- Agent YAML files (e.g., `configs/agent/latent_attn_tiny.yaml`) are loaded locally
- Merged into the training config before serialization
- No need for remote deserialization

### 3. YAML Transfer
- Complete config serialized to `/tmp/metta_configs/train_config_*.yaml`
- Transferred via Skypilot file_mounts to `/tmp/metta_train_config.yaml`
- Remote uses: `--config-path=/tmp --config-name=metta_train_config`

## Testing Commands

### 1. Test In-Memory Config Updates

```python
# Test that additional_args modify the config in-memory
from experiments.training_run_config import TrainingRunConfig

config = TrainingRunConfig(
    additional_args=[
        "trainer.total_timesteps=50000",
        "trainer.optimizer.learning_rate=0.001",
        "trainer.ppo.clip_coef=0.3",
    ]
)

# Apply the args to update the in-memory config
config.apply_additional_args()

# Verify the config was updated
assert config.trainer.total_timesteps == 50000
assert config.trainer.optimizer.learning_rate == 0.001
assert config.trainer.ppo.clip_coef == 0.3

# Serialize to YAML - should contain the updated values
yaml_dict = config.serialize_to_yaml()
assert yaml_dict["trainer"]["total_timesteps"] == 50000
```

### 2. Test Agent Config Loading

```python
# Test that agent configs are loaded and merged
from experiments.training_run_config import TrainingRunConfig

config = TrainingRunConfig(
    agent_config="latent_attn_tiny",
)

# Load and merge agent config
config.load_agent_config()

# The agent config should be merged into the training config
yaml_dict = config.serialize_to_yaml()
# Should have agent-specific settings merged in
assert "agent" in yaml_dict
assert yaml_dict["agent"]["architecture"] == "latent_attn_tiny"
```

### 3. Test Complete Workflow

```bash
# Create experiment with additional args
uv run experiments/recipes/arena_experiment.py \
  --no-launch \
  --total-timesteps=100000 \
  --learning-rate=0.0005 \
  --batch-size=1024 \
  test_complete

# This should:
# 1. Create TrainingRunConfig
# 2. Apply the overrides to the in-memory config
# 3. Load and merge agent config
# 4. Serialize complete YAML
# 5. Show the YAML path without launching
```

### 4. Verify YAML is Self-Contained

```bash
# Generate a config
uv run experiments/recipes/arena_experiment.py --no-launch test_yaml

# Get the YAML path from output
YAML_PATH="/tmp/metta_configs/train_config_*.yaml"

# The YAML should be completely self-contained
cat $YAML_PATH

# Expected structure:
# - Complete trainer config with all settings
# - Agent config merged in
# - No references to external files
# - No need for additional command-line args
```

### 5. Test Skypilot Integration

```bash
# Launch with config
uv run experiments/recipes/arena_experiment.py \
  --gpus=1 \
  --total-timesteps=10000 \
  sky_test_$(date +%s)

# SSH into the job
sky exec $JOB_NAME bash

# Verify the complete config was transferred
cat /tmp/metta_train_config.yaml

# Should contain all settings, fully resolved
# No additional hydra composition needed
```

## Implementation Details

### apply_additional_args() Method

```python
def apply_additional_args(self):
    """Apply additional_args to update the in-memory config."""
    if not self.additional_args:
        return
    
    for arg in self.additional_args:
        if "=" not in arg:
            continue
        
        key, value = arg.split("=", 1)
        keys = key.split(".")
        
        # Navigate to the target object
        obj = self
        for k in keys[:-1]:
            obj = getattr(obj, k)
        
        # Set the value with type conversion
        setattr(obj, keys[-1], self._convert_value(value))
```

### load_agent_config() Method

```python
def load_agent_config(self):
    """Load agent YAML and merge into config."""
    agent_path = Path("configs/agent") / f"{self.agent_config}.yaml"
    
    with open(agent_path) as f:
        agent_dict = yaml.safe_load(f)
    
    # Merge agent settings into trainer config
    if not self.trainer:
        self.trainer = self.get_trainer_config()
    
    # Apply agent-specific overrides
    self.trainer.update_from_dict(agent_dict)
```

## Benefits of This Approach

1. **Type Safety**: All config updates happen on typed Python objects
2. **Validation**: Pydantic validates all settings before serialization
3. **Completeness**: YAML contains everything needed, no remote composition
4. **Debuggability**: Can inspect the exact YAML being sent
5. **Flexibility**: Easy to modify configs programmatically
6. **No Command-Line Limits**: Everything in the file, not on command line

## Migration Notes

### From Old System
```python
# Old: Pass additional_args to command line
additional_args=["trainer.learning_rate=0.001"]
# These were passed as: train.py trainer.learning_rate=0.001
```

### To New System
```python
# New: Apply to in-memory config
config = TrainingRunConfig(
    additional_args=["trainer.learning_rate=0.001"]
)
config.apply_additional_args()
# Config object now has trainer.optimizer.learning_rate = 0.001
# YAML will contain the complete, updated config
```

## Testing Checklist

- [ ] additional_args modify in-memory configs
- [ ] Agent configs are loaded and merged locally
- [ ] YAML is self-contained with no external references
- [ ] Skypilot receives complete config via file_mounts
- [ ] No command-line args needed for config on remote
- [ ] Type validation happens before serialization
- [ ] Preview mode shows exact YAML that will be used