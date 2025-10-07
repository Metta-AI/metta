# AGENTS.md

Codex agents working in this repository should follow these guidelines:

## Commit messages

- Use Conventional Commits where reasonable (`feat:`, `fix:`, `refactor:`, `docs:`, etc.)
- Keep them short and present tense
- Describe the change clearly

## Quality checks

Before finalizing changes (e.g. before a commit), or upon request from the user:

- Run `ruff format` and `ruff check` on Python files before committing.
- Run the unit tests with `uv run pytest` or by activating the venv and running `pytest`.

## Type Annotations

- Always add type annotations to function parameters
- Add return type annotations to public API functions
- Follow selective annotation guidelines (see CLAUDE.md for details)
- Run mypy to check type consistency before committing

## Pull request notes

- Mention relevant file paths when describing changes.
- Include test output or note why tests were skipped.

## Experiments and long-running tools

- When running ABES recipes (e.g. `./tools/run.py experiments.recipes.abes.vit.train`), wrap the invocation in a shell `timeout` so local runs terminate cleanly. Example: `timeout 30s ./tools/run.py experiments.recipes.abes.vit.train run=local_vit trainer.total_timesteps=5_000_000`.
- For quick end-to-end CoGames checks from the package directory, run `timeout 30s uv run cogames train training_facility_1 --policy simple` (from within `packages/cogames/`).
