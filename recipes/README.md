# Metta Training Recipes

This directory contains standardized training recipes for Metta that can be easily run and automated as part of continuous integration.

## Quick Start

### List available recipes
```bash
python recipes/run_recipe.py --list
```

### Run a recipe
```bash
python recipes/run_recipe.py navigation
python recipes/run_recipe.py object_use
python recipes/run_recipe.py nav_memory_sequence
```

### Using Makefile (easier)
```bash
cd recipes
make help                    # Show all available commands
make list                    # List recipes
make run-navigation          # Run navigation recipe
make run-all                 # Run all stable recipes
make validate                # Validate all configurations
```

## Recipe Structure

Each recipe is defined in a YAML file with the following structure:

```yaml
name: "Recipe Name"
description: "What this recipe does"
baseline_run: "https://wandb.ai/.../runs/..."
expected_performance: "Expected performance metrics"

# Training configuration
run_suffix: "recipe.baseline"
trainer: "recipe_trainer"
curriculum: "env/mettagrid/curriculum/..."

# Environment overrides
env_overrides:
  game.num_agents: 4

# CI/Testing configuration
ci_config:
  timeout_hours: 24
  expected_min_performance: 0.85
  validation_metrics:
    - "success_rate"
```

## Available Recipes

### Navigation Recipe
- **File**: `navigation.yaml`
- **Expected Performance**: >88% on navigation evals
- **Baseline**: [daphne.navigation.low_reward.1gpu.4agents.06-25](https://wandb.ai/metta-research/metta/runs/daphne.navigation.low_reward.1gpu.4agents.06-25)

### Object Use Recipe
- **File**: `object_use.yaml`
- **Expected Performance**: ~60% on object use evals
- **Baseline**: [daphne.object_use.06-25](https://wandb.ai/metta-research/metta/runs/daphne.object_use.06-25?nw=nwuserdaphned)

### Navigation Memory Sequence Recipe
- **File**: `nav_memory_sequence.yaml`
- **Status**: Experimental
- **Variants**:
  - Object use finetuned
  - Navigation finetuned

## CI Integration

The recipes are automatically tested as part of the CI pipeline:

- **Pull Requests**: Configuration validation and short tests
- **Main Branch**: Full recipe execution
- **Weekly**: Scheduled runs including experimental recipes

### Manual CI Trigger
You can manually trigger recipe tests in GitHub Actions:
1. Go to Actions → Recipe Tests
2. Click "Run workflow"
3. Optionally specify a specific recipe

## Adding New Recipes

1. Create a new YAML file in the `recipes/` directory
2. Follow the structure shown above
3. Add the recipe to the CI workflow if needed
4. Update this README

## Troubleshooting

### Common Issues

1. **Recipe not found**: Make sure the YAML file exists and has the correct name
2. **Configuration errors**: Run `make validate` to check all configurations
3. **Timeout issues**: Adjust `timeout_hours` in the recipe configuration

### Debug Mode
```bash
python recipes/run_recipe.py <recipe> --dry-run
```

This will show what command would be executed without actually running it.

## Migration from Bash Scripts

The old bash scripts (`*.sh`) are being replaced with YAML configurations. The new system provides:

- **Better error handling**
- **CI integration**
- **Standardized configuration**
- **Easier maintenance**
- **Validation capabilities**

To migrate an existing bash script:
1. Extract the configuration parameters
2. Create a corresponding YAML file
3. Test with `--dry-run`
4. Update any documentation
