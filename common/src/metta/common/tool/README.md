# Tool Runner (./tools/run.py)

The tool runner loads a Tool from a recipe module, applies CLI overrides, and invokes it. It supports convenient two-token shorthand and recipe inference.

## Quick Start

```bash
# Train (arena)
./tools/run.py train arena run=my_experiment

# Evaluate a policy
./tools/run.py evaluate arena policy_uri=s3://my-bucket/checkpoints/run/run:v12.pt

# Interactive play (browser)
./tools/run.py play arena policy_uri=file://./train_dir/run/checkpoints/run:v12.pt

# Generate & view a replay
./tools/run.py replay arena policy_uri=file://./train_dir/run/checkpoints/run:v12.pt
```

## Two-Token Form and Shorthands

- Two-token: `./tools/run.py <tool> <recipe>` is equivalent to `<recipe>.<tool>`.
- Omit prefix: `arena` maps to `experiments.recipes.arena`.

Examples:

```bash
# Equivalent invocations
./tools/run.py evaluate arena policy_uri=...
./tools/run.py arena.evaluate policy_uri=...
```

## Listing Tools

```bash
# List explicit tools defined by a recipe module
./tools/run.py train arena --list-tools

# Equivalent (uses the same module resolution):
./tools/run.py arena.train --list-tools
```

- Shows tools exported by that recipe (functions/classes returning `Tool`).
- Inferred tools (constructed from `mettagrid()`/`simulations()`) are not listed here, but they still work when invoked.

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

## Tips

- Strings with spaces should be quoted: `notes="my run"`.
- Booleans are lowercase: `true`, `false`.
- If a numeric-looking value should be a string, quote it (e.g., `run="001"`).

