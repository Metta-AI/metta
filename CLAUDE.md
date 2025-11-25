# CLAUDE.md

Guidance for AI assistants working on this codebase.

## Setup

```bash
./install.sh              # Initial setup
metta status              # Check component status
metta install             # Reinstall if imports fail
```

## Common Commands

```bash
# Training (use --timeout with long-running commands)
uv run ./tools/run.py train arena run=my_experiment

# Evaluation
uv run ./tools/run.py evaluate arena policy_uri=file://./train_dir/my_run/checkpoints

# List available tools
uv run ./tools/run.py arena --list

# Testing (only run when specifically needed - takes minutes)
metta pytest tests/path/to/test.py -v    # Specific test
metta ci                                  # Full CI suite

# Linting (only run when specifically needed)
metta lint path/to/file.py --fix
```

## Repository Structure

```
metta/          # Private - core RL training, not published separately
packages/
  mettagrid/    # Public - C++/Python grid environment
  cogames/      # Public - game configs, depends on mettagrid
recipes/
  prod/         # Production recipes with CI validation
  experiment/   # Work-in-progress recipes
```

Dependency direction: `metta` depends on `cogames` depends on `mettagrid`. Nothing depends on `metta`.

## Recipe System

Recipes define tool functions that configure training/evaluation:

```bash
./tools/run.py train arena run=test           # Two-token form
./tools/run.py arena.train run=test           # Dot notation
./tools/run.py arena --list                   # Show available tools
```

See `common/src/metta/common/tool/README.md` for details.

## Code Style

- Type hints on all function parameters
- No docstrings or comments unless logic is non-obvious
- Absolute imports, not relative
- Private members start with `_`
- Empty `__init__.py` files
- `uv run` for all Python commands

### Imports

```python
from __future__ import annotations  # When needed for forward refs
from metta.common.types import X    # Shared types from types.py files
```

If circular import: extract types to `types.py` or use module import (`import x.y as y_mod`).

## Working Style

**Keep it simple.** Write less code that will be kept, not more code that shows an MVP. Look for existing implementations before writing new ones.

**Don't run lint/tests automatically.** They take too long. Only run when specifically needed or requested.

**Don't interact with git remote.** No pushing, no PRs. Humans handle that.

**Use timeouts for long commands.** Training and evaluation can run indefinitely:
```bash
# Bad - may hang
uv run ./tools/run.py train arena run=test

# Better - will timeout
uv run ./tools/run.py train arena run=test trainer.total_timesteps=100000
```

**Search before writing.** The codebase likely has something similar to what you need.

## Testing

Tests in `tests/` mirror the source structure. Run specific tests, not the full suite:

```bash
metta pytest tests/rl/test_trainer_config.py -v
```

## Configuration

OmegaConf configs in `configs/`:
- `agent/` - Neural network architectures
- `trainer/` - Training hyperparameters
- `sim/` - Simulation/evaluation configs
