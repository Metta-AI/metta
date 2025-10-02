# Tool Runner (./tools/run.py)

The tool system is built on two core concepts:

## Architecture: Tools and Recipes

### Tools

**Tools** are runnable units of work defined as Pydantic config classes that inherit from `Tool`. Each tool type has:
- A `tool_name` class variable (e.g., "train", "eval", "play")
- Configuration fields defining its behavior and input
- An `invoke()` method that executes the tool

Common tool types:
- `TrainTool` - Training an agent
- `EvalTool` - Evaluating policies
- `PlayTool` - Interactive browser-based play
- `ReplayTool` - Generating replay videos
- `AnalysisTool` - Analyzing evaluation results

Tools are defined in `metta/tools/` and provide the interface for all operations.

### Recipes

**Recipes** are Python modules (typically in `experiments/recipes/`) that define **tool functions** - functions that return configured Tool instances for specific experiments or environments.

**Key benefit**: Recipes group related configurations together. For example, an `arena` recipe ensures you train, evaluate, and play on the same maps and configurations - maintaining consistency across your entire workflow.

A recipe module contains:
- **Tool functions**: Functions with return type annotations that return Tool instances (e.g., `def train() -> TrainTool`)
- **Helper functions** (optional): Shared config like `mettagrid()` or `simulations()` to avoid duplication across tools

TODO: Make Recipe subclasses or otherwise allow you to get many tool definitions from a few core such functions.
Jack previously explored this and found the code complexity to benefit ratio wasn't there, but it could be if
we make Recipes first-class objects and/or discover specific patterns that are worth "automating".

Example recipe structure:
```python
# experiments/recipes/arena.py
from metta.tools.train import TrainTool
from metta.tools.eval import EvalTool
from metta.tools.play import PlayTool

def simulations():
    """Shared arena simulations used across tools."""
    return [
        SimulationConfig(suite="arena", name="basic", env=make_arena()),
        SimulationConfig(suite="arena", name="combat", env=make_arena_combat()),
    ]

def train() -> TrainTool:
    """Train on arena maps."""
    return TrainTool(
        training_env=...,
        evaluator=EvaluatorConfig(simulations=simulations())  # Same maps for eval
    )

def eval() -> EvalTool:
    """Evaluate on arena maps (same as training)."""
    return EvalTool(simulations=simulations())

def play() -> PlayTool:
    """Interactive play on arena maps."""
    return PlayTool(sim=simulations()[0])
```

This ensures training, evaluation, and play all use the same arena configurations.

### How They Work Together

1. **Recipe Discovery**: The system automatically discovers recipes in `experiments/recipes/`
2. **Tool Loading**: When you run `./tools/run.py train arena`, the runner:
   - Finds the `arena` recipe module
   - Locates the `train` function (tool function)
   - Calls `train()` to get a configured `TrainTool` instance
   - Applies CLI argument overrides to the tool config
   - Calls `tool.invoke()` to execute
3. **Layering**: Recipes depend on Tools (not vice versa) - tools are reusable "verbs", recipes are "nouns" that configure them for specific use cases

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
