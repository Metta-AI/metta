# Recipe Validation

Validates training and cluster workflows locally and remotely for stable releases.

## Quick Start

```bash
# Validate training (1 local + 1 remote)
./devops/stable/recipe_validation/validate_recipes.py launch train

# Validate cluster exit conditions (1 remote)
./devops/stable/recipe_validation/validate_recipes.py launch cluster

# Check results
./devops/stable/recipe_validation/validate_recipes.py check

# Check with logs
./devops/stable/recipe_validation/validate_recipes.py check -l
```

## What Gets Validated

### Training (`train`)
- **arena_local**: 1k timesteps locally (~1 min) - fast smoke test
- **arena_remote**: 50k timesteps on cluster (~10 min) - full validation

### Cluster Config (`cluster`)
- **runtime_timeout**: Verifies timeout handling works correctly

## How It Works

**Local validations** run via `uv run ./tools/run.py` with limited timesteps. Fast feedback.

**Remote validations** launch SkyPilot jobs and track them via job IDs. Results saved to JSON.

Success = exit code 0. Failures show in red with error details.

## Commands

### Launch
```bash
./devops/stable/recipe_validation/validate_recipes.py launch <tool>
```
- `--output-file`: Custom JSON output file
- `--skip-git-check`: Skip git validation

### Check
```bash
./devops/stable/recipe_validation/validate_recipes.py check
```
- `-f/--input-file`: Specify JSON file (auto-detected by default)
- `-l/--logs`: Show detailed logs
- `-n/--tail-lines`: Lines of logs to show (default 200)

## Output

Saves to `<tool>_validation_jobs.json`:
```json
{
  "test_run_info": { "base_name": "train_validation", ... },
  "launched_jobs": [
    { "job_id": "123", "recipe": "arena_remote", "status": "running" }
  ]
}
```

Local results stored inline. Remote jobs tracked by job ID.

## Adding Validations

Edit `TOOL_VALIDATIONS` in `recipe_validator.py`:

```python
RecipeValidation(
    name="my_recipe",
    module="experiments.recipes.my_recipe.train",
    description="What this tests",
    location=ValidationLocation.LOCAL,  # or REMOTE
    condition=TestCondition(
        name="Quick Test",
        extra_args=["trainer.total_timesteps=1000"],
        description="Brief description",
        ci=True,
    ),
)
```

Keep it minimal. Add validations as needed, not preemptively.
