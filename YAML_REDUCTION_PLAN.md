# YAML Reduction Plan

This document outlines which YAML files can be reduced or removed as part of the Metta refactoring.

## Files That Can Be Removed

### 1. Trainer Configs (`configs/trainer/*.yaml`)
**Replace with:** `configs/python/training_configs.py`

These files define training hyperparameters and can be replaced with Python dataclasses:
- `simple.yaml`
- `puffer.yaml`
- `trainer.yaml`
- `simple.medium.yaml`
- `g16.yaml`

### 2. Hardware Configs (`configs/hardware/*.yaml`)
**Replace with:** Hardware-specific functions in `training_configs.py`

- `mac_serial.yaml`
- `mac_parallel.yaml`
- `macbook.yaml`
- `pufferbox.yaml`
- `aws.yaml`
- `github.yaml`

### 3. User Configs (`configs/user/*.yaml`)
**Replace with:** User-specific Python scripts or environment variables

All user configs can be removed as users should create their own training scripts.

### 4. Top-level Job Configs
**Replace with:** Direct Python scripts in `tools/`

- `train_job.yaml` → `tools/train_new.py`
- `sim_job.yaml` → `tools/evaluate_new.py`
- `play_job.yaml` → `tools/play_new.py`
- `renderer_job.yaml` → Integrated into `play_new.py`

## Files That Should Be Simplified

### 1. Agent Configs (`configs/agent/*.yaml`)
**Replace with:** `configs/python/agents.py`

Convert agent architectures to Python functions that return component dictionaries:
- `simple.yaml` → `simple_cnn_agent()`
- `latent_attn.yaml` → `attention_agent()`
- etc.

### 2. Environment Configs (`configs/env/mettagrid/*`)
**Replace with:** `configs/python/environments.py`

The deeply nested environment configs can be replaced with Python classes:
- Navigation environments → `NavigationWalls`, `NavigationMaze`, etc.
- Memory environments → `MemorySequence`, `MemoryLandmarks`, etc.
- Object use environments → `ObjectUseArmory`, etc.

### 3. Simulation Configs (`configs/sim/*.yaml`)
**Replace with:** Python functions that return environment suites

- `all.yaml` → `all_eval_suite()`
- `navigation.yaml` → `navigation_eval_suite()`
- `memory.yaml` → `memory_eval_suite()`

## Files That Should Remain (For Now)

### 1. Core Infrastructure
- `hydra.yaml` - Keep for backward compatibility
- `common.yaml` - May still be useful for shared settings

### 2. Complex Configurations
Some complex game configurations might be better kept as YAML/JSON data files rather than Python code, especially if they contain large amounts of data or are frequently edited by non-programmers.

## Migration Strategy

### Phase 1: Add Python Alternatives (DONE)
✅ Create Python-based configuration system
✅ Create new tools that use Python configs
✅ Maintain backward compatibility

### Phase 2: Gradual Migration
- Update documentation to prefer Python configs
- Convert examples to use new system
- Deprecate YAML-based tools

### Phase 3: Cleanup
- Remove unused YAML files
- Move remaining YAML files to a `legacy/` directory
- Update all tests to use Python configs

## Benefits of Reduction

1. **Fewer files to maintain** - From 200+ YAML files to ~10 Python modules
2. **Better IDE support** - Autocomplete, type checking, refactoring tools
3. **Easier to understand** - Configuration logic is in code, not in override chains
4. **More flexible** - Can use loops, conditions, and functions in configs
5. **Easier testing** - Standard Python testing tools work

## Example: Before and After

### Before (YAML + Hydra)
```bash
python tools/train.py \
  +trainer=puffer \
  +agent=simple \
  +env=mettagrid/navigation/training/varied_terrain_sparse \
  +hardware=gpu_4x \
  trainer.total_timesteps=100000000
```

### After (Python)
```python
from configs.python.training_configs import large_scale_config, ppo_default
from configs.python.agents import simple_cnn_agent
from configs.python.environments import NavigationWalls

config = large_scale_config()
config.total_timesteps = 100_000_000

agent_config = simple_cnn_agent()
env = NavigationWalls(width=50, height=50)

# Train with explicit configuration
train(agent_config, env, config, ppo_default())
```

The Python approach is more verbose but much clearer about what's actually happening.
