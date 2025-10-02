# Tool Runner (./tools/run.py)

The tool runner loads a Tool from a recipe module, applies CLI overrides, and invokes it. It supports convenient two-token shorthand and recipe inference.

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

# Evaluate a policy
./tools/run.py evaluate arena policy_uri=file://./train_dir/my_run/checkpoints

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
./tools/run.py evaluate --list       # Shows all recipes with evaluate tools
```

The output shows both:
- **Explicit tools**: Functions/classes that return a Tool (e.g., `train_shaped`)
- **Inferred tools**: Automatically generated from `mettagrid()`/`simulations()` (e.g., `train`, `evaluate`, `play`, `replay`)

## Dry-Run

```bash
# Validate resolution only (does not construct or run the tool)
./tools/run.py evaluate arena --dry-run
```

## Recipe Inference

Recipes can expose either of the following:

```python
def mettagrid() -> MettaGridConfig: ...
def simulations() -> list[SimulationConfig]: ...
```

When present, the runner can infer common tools even if the module does not export them explicitly:

- `train`: requires `mettagrid()`. If `simulations()` is present, it is used for the evaluator in training.
- `play`: uses `simulations()[0]` when available; otherwise falls back to `mettagrid()`.
- `replay`: uses `simulations()[0]` when available; otherwise falls back to `mettagrid()`.
- `evaluate`/`evaluate_remote`: use `simulations()` if present; otherwise wrap `mettagrid()` as a single simulation.

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
