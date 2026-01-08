# CLAUDE.md

Guidance for AI assistants working on this codebase. See also `STYLE_GUIDE.md`.

## Setup

```bash
./install.sh              # Initial setup
metta status              # Check component status
metta install             # Reinstall if imports fail
```

Most of the time, you shouldn't need to run install.sh or metta install. Only run these if you're having trouble with
imports or other setup issues.

## Commands

```bash
# Training (always use timestep limit to avoid hanging)
uv run ./tools/run.py train arena run=my_experiment trainer.total_timesteps=100000

# Evaluation
uv run ./tools/run.py evaluate arena policy_uri=file://./train_dir/my_run/checkpoints

# List available tools
uv run ./tools/run.py arena --list
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

## Package Dependencies

```
metta/ ──────► cogames/ ──────► mettagrid/
app_backend/ ──► common/ (only)
```

- Nothing depends on `metta/` (it's the top-level consumer)
- `mettagrid` has no internal Python dependencies (C++/Python hybrid)
- `app_backend` is isolated, can only import from `common/`
- Enforced by `import-linter`. Run `uv run lint-imports` to check. See `.importlinter`.

## Testing

- Don't speculatively run tests; only run when asked or in a targeted way for changes you made and want to validate

```bash
metta pytest tests/path/to/test.py -v    # Run specific test
metta pytest --changed                    # Run only tests affected by your changes
```

## Proto Files

Files in `proto/` define schemas for cross-system boundaries (network APIs, files on disk). Do not modify proto schemas
as part of other refactoring work. Schema changes require explicit discussion because they can break compatibility with
files written using older schemas or services that haven't been redeployed.

## Recipe System

```bash
./tools/run.py train arena run=test           # Two-token form
./tools/run.py arena --list                   # Show available tools
```

See `common/src/metta/common/tool/README.md` for details.

## Git Hub Integration

Use graphite ("gt") to create PRs. Name the branch $user-short-issue-name (5 words or less)
