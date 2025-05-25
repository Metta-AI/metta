# AGENTS.md

Codex agents working in this repository should follow these guidelines:

## Commit messages
- Keep them short and present tense.
- Describe the change clearly.

## Quality checks
- Run `ruff format` and `ruff check` on Python files before committing.
- Run the unit tests with `uv run pytest` or by activating the venv and running `pytest`.

## Pull request notes
- Mention relevant file paths when describing changes.
- Include test output or note why tests were skipped.
