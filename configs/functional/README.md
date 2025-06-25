# Functional Training Configuration

This directory contains structured configuration files that can be used with the functional training demo (`demo.py`). These configs are organized by concern to make training configuration more modular and reusable.

## Configuration Structure

- **ppo/default.yaml** - PPO algorithm parameters (gamma, GAE, clipping, etc.)
- **optimizer/adam.yaml** - Adam optimizer settings
- **optimizer/muon.yaml** - Muon optimizer settings (experimental)
- **training/default.yaml** - Training loop parameters (batch sizes, timesteps, etc.)
- **environment/default.yaml** - Environment and worker settings
- **kickstarter/default.yaml** - Imitation learning/kickstart configuration

## Usage with demo.py

The functional training demo (`demo.py`) can use these configs in two ways:

### 1. Default Pydantic Configuration (Built-in Defaults)
```python
# Uses sensible defaults defined in Pydantic models
ppo_config = PPOConfig()
optimizer_config = OptimizerConfig()
training_config = TrainingConfig()
env_config = EnvironmentConfig()
```

### 2. Load from YAML Files
```python
# Load explicit configurations from YAML
ppo_config = PPOConfig.from_yaml("configs/ppo/default.yaml")
optimizer_config = OptimizerConfig.from_yaml("configs/optimizer/adam.yaml")
training_config = TrainingConfig.from_yaml("configs/training/default.yaml")
env_config = EnvironmentConfig.from_yaml("configs/environment/default.yaml")
```

## Relationship to Existing Training

These configs exist alongside the existing Hydra-based training system. The main training script (`tools/train.py`) continues to use the original trainer configurations in `configs/trainer/`.

This functional approach provides:
- Clear separation of concerns
- Easy to understand configuration without Hydra complexity
- Direct object creation without framework magic
- Same training capabilities in a simpler interface

## Running the Demo

```bash
# Run with default configs
python demo.py

# Or modify demo.py to load from YAML files
# (uncomment the YAML loading section in main())
```

The demo will create all training objects explicitly and run a simple while loop calling `rollout()` and `train_ppo()` functions.
