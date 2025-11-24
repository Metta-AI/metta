# AGENTS.md

Codex agents working in this repository should follow these guidelines:

## Commit messages

- Use Conventional Commits where reasonable (`feat:`, `fix:`, `refactor:`, `docs:`, etc.)
- Keep them short and present tense
- Describe the change clearly

## Quality checks

Before finalizing changes (e.g. before a commit), or upon request from the user:

- Run `metta lint` (or `metta lint --fix`) on the files you touched. This covers Ruff plus all Prettier-backed formats.
- Run the unit tests with `uv run pytest` or by activating the venv and running `pytest`.
- When pruning unused code, confirm it is referenced by production codepaths. If a symbol is only covered (or mentioned)
  in tests/fixtures, treat that as delete-worthy cruft and remove it rather than keeping legacy shims, unless the user
  explicitly requests otherwise.

## Import Patterns

**Core rules:**

- **Always use absolute imports**, not relative imports
  - CORRECT: `from metta.rl.trainer import Trainer`
  - INCORRECT: `from .trainer import Trainer`
- **Use `from __future__ import annotations`** when needed for forward references (e.g., methods returning their own
  class type, or type hints referencing classes defined later in the file)
- **Use module imports** to break circular dependencies: `import metta.rl.trainer as trainer`
- **Check `pyproject.toml`** dependencies - only import from packages in your dependency tree

**When you hit a circular import:**

1. Extract shared types to `types.py` at the lowest common package
2. Convert to module imports: `import X as X_mod` instead of `from X import Y`
3. If stuck, ask for review

**Performance exceptions:**

- Public packages (`packages/`): Use lazy loading via `__getattr__` for heavy imports (torch, gymnasium, etc.)
- Internal modules (`metta/`): Keep `__init__.py` files empty or minimal
- Module-level `__getattr__` allowed in regular `.py` files when documented performance benefit exists

**Not allowed:**

- Inline imports (imports inside function/method bodies) to work around circular dependencies
- Use the resolution protocol above instead

See `tools/dev/python_imports/SPECIFICATION.md` for detailed guidance

## Type Annotations

- Always add type annotations to function parameters
- Add return type annotations to public API functions
- Use `Optional[type]` instead of `type | None` for optional types
- Follow selective annotation guidelines (see CLAUDE.md for details)
- Run mypy to check type consistency before committing

## Pull request notes

- Mention relevant file paths when describing changes.
- Include test output or note why tests were skipped.

## Experiments and long-running tools

- When running ABES recipes (e.g. `./tools/run.py abes.vit.train`), wrap the invocation in a shell `timeout` so local
  runs terminate cleanly. Example:
  `timeout 30s ./tools/run.py abes.vit.train run=local_vit trainer.total_timesteps=5_000_000`.
- For quick end-to-end CoGames checks from the package directory, run
  `timeout 30s uv run cogames train training_facility_1 --policy simple` (from within `packages/cogames/`).
