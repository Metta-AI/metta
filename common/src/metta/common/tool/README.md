# Tool Runner (./tools/run.py)

The tool runner loads a Tool from a recipe module, applies CLI overrides, and invokes it. It supports convenient two-token shorthand.

## Recipe Discovery

Recipes are automatically discovered from the `experiments/recipes` directory using `pkgutil.walk_packages()`.

**Important**: Python requires subdirectories to have `__init__.py` files to be importable as packages. If you create a new subdirectory for recipes, run:

```bash
uv run python devops/tools/ensure_recipe_packages.py
```

This automatically creates empty `__init__.py` files in any subdirectories that need them.

## Quick Start

```bash
# Train (arena)
./tools/run.py train arena run=my_experiment

# Eval a policy
./tools/run.py eval arena policy_uri=file://./train_dir/my_run/checkpoints

# Interactive play (browser)
./tools/run.py play arena policy_uri=file://./train_dir/my_run/checkpoints

# Generate & view a replay
./tools/run.py replay arena policy_uri=file://./train_dir/my_run/checkpoints
```

## Two-Token Form and Shorthands

The runner supports flexible syntax for invoking tools:

**Two-token form**: `./tools/run.py <tool> <recipe>` is equivalent to `<recipe>.<tool>`
- Example: `train arena` → `arena.train`

**Short recipe names**: Omit the `experiments.recipes.` prefix
- Example: `arena` → `experiments.recipes.arena`

**Equivalent invocations**:
```bash
./tools/run.py train arena run=test
./tools/run.py arena.train run=test
./tools/run.py experiments.recipes.arena.train run=test
```

## Discovering Tools

Use `--list` to discover available tools:

```bash
# List all tools in a specific recipe
./tools/run.py arena --list
./tools/run.py navigation --list

# List all recipes that provide a specific tool
./tools/run.py train --list          # Shows all recipes with train tools
./tools/run.py eval --list           # Shows all recipes with eval tools
```

The output shows all available tool functions that return Tool instances (e.g., `train`, `eval`, `train_shaped`).

## Dry-Run

```bash
# Validate resolution only (does not construct or run the tool)
./tools/run.py eval arena --dry-run
```

## Recipe Structure

Recipes define explicit tool functions that return Tool instances:

```python
from metta.tools.train import TrainTool
from metta.tools.eval import EvalTool

def train() -> TrainTool:
    return TrainTool(...)

def eval() -> EvalTool:
    return EvalTool(simulations=[...])
```

Recipes can optionally define helper functions like `mettagrid()` or `simulations()` to avoid duplication when multiple tools need the same configuration.

## Argument Classification

```bash
# Function args (bound by name) and nested overrides
./tools/run.py train arena \
  run=local.alice.1 \
  system.device=cpu \
  wandb.enabled=false \
  trainer.total_timesteps=100000

# Show classification with --verbose
./tools/run.py train arena run=test --verbose
```
