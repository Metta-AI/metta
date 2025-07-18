# AGENTS.md

Codex agents working in this repository should follow these guidelines:

## Commit messages

- Keep them short and present tense.
- Describe the change clearly.

## Quality checks

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