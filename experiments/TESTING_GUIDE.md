# Testing Guide - Arena Experiment

This guide shows how to test the Arena experiment with the YAML serialization system.

## Quick Start

### 1. Preview Mode (No Launch)

Test the Arena experiment locally without launching to see what will be generated:

```bash
# Basic preview - shows YAML without launching
uv run experiments/recipes/arena_experiment.py --no-launch

# Preview with different infrastructure settings
uv run experiments/recipes/arena_experiment.py \
  --no-launch \
  --gpus 4 \
  --nodes 2 \
  --no-spot
```

This will:
- Generate the YAML configuration with arena curriculum
- Display the config details that will be sent
- Show local testing commands with `tools/train.py`
- Create YAML files in `/tmp/metta_configs/` and `~/.metta_test_configs/`
- NOT launch to Skypilot

### 2. Test Generated YAML Locally

The preview mode provides a command to test with tools/train.py:

```bash
# Run preview mode first
uv run experiments/recipes/arena_experiment.py --no-launch

# Look for output like:
# To test locally with tools/train.py:
#   uv run ./tools/train.py --config-path=/Users/you/.metta_test_configs --config-name=test_config_20250113_123456

# View the generated config (shows what Hydra will compose)
uv run ./tools/train.py \
  --config-path=/Users/you/.metta_test_configs \
  --config-name=test_config_20250113_123456 \
  --cfg job
```

### 3. Launch Arena Training to Skypilot

When ready to launch:

```bash
# Basic launch with defaults
uv run experiments/recipes/arena_experiment.py

# Launch with custom name and infrastructure
uv run experiments/recipes/arena_experiment.py \
  --name my_arena_run \
  --gpus 2 \
  --nodes 1
```

## Verifying the Configuration

### Check Generated YAML

```bash
# Find generated YAML files
ls -la ~/.metta_test_configs/test_config_*.yaml

# Inspect the content
cat ~/.metta_test_configs/test_config_*.yaml | head -20
```

Expected content:
```yaml
# @package _global_
defaults:
- common
- 'agent: fast'
- 'trainer: trainer'
- 'sim: arena'
- 'wandb: metta_research'
- _self_
seed: 1
trainer:
  curriculum: env/mettagrid/curriculum/arena/learning_progress
  ...
```

## Monitoring Skypilot Jobs

After launching:

```bash
# Check job status
sky jobs queue

# View logs (replace JOB_ID with actual ID)
sky jobs logs JOB_ID

# Check the transferred config
sky exec JOB_ID 'cat /tmp/metta_train_config.yaml | head -20'

# Verify train.py is using the config
sky exec JOB_ID 'ps aux | grep train.py'
```

## Creating Custom Experiments

To create your own experiment with different settings:

1. Copy `experiments/recipes/arena_experiment.py` to a new file
2. Modify the curriculum and other defaults:

```python
class MyExperimentConfig(SingleJobExperimentConfig):
    name: str = "my_experiment"
    
    def __init__(self, **kwargs):
        if 'training' not in kwargs:
            kwargs['training'] = TrainingRunConfig(
                curriculum="env/mettagrid/curriculum/navigation/basic",
                agent_config="latent_attn_tiny",
                wandb_tags=["custom", "navigation"],
            )
        super().__init__(**kwargs)
```

## Troubleshooting

### Common Issues

1. **ValidationError: curriculum Field required**
   - Experiments must specify a curriculum
   - Arena experiment sets: `env/mettagrid/curriculum/arena/learning_progress`

2. **"Could not load 'wandb: metta_research'"**
   - Expected when testing locally with custom config paths
   - The remote environment has all configs available

3. **Skypilot launch fails**
   - Check git state: `git status` (should be clean)
   - Verify AWS/wandb credentials are set
   - Ensure you're in the metta repository root

### Debug Commands

```bash
# Test arena experiment config creation
uv run python -c "
from experiments.recipes.arena_experiment import ArenaExperimentConfig
config = ArenaExperimentConfig(name='test')
print(f'Curriculum: {config.training.curriculum}')
print(f'Agent: {config.training.agent_config}')
"

# Test YAML serialization
uv run python -c "
from experiments.recipes.arena_experiment import ArenaExperimentConfig
config = ArenaExperimentConfig(name='test')
path, yaml_dict = config.training.serialize_to_yaml_file()
print(f'Created: {path}')
import yaml
with open(path) as f:
    content = yaml.safe_load(f)
print(f'Curriculum in YAML: {content[\"trainer\"][\"curriculum\"]}')
"
```

## Running Tests

```bash
# Run all experiments tests
uv run pytest tests/experiments/ -v

# Run specific test files
uv run pytest tests/experiments/test_yaml_contract.py -v
uv run pytest tests/experiments/test_train_integration.py -v
uv run pytest tests/experiments/test_experiment_workflows.py -v
```